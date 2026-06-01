
import camb
import numpy as np


def get_bb_cls(r=0.01, lmax=300):

    pars = camb.CAMBparams()

    pars.set_cosmology(
        H0=67.36,
        ombh2=0.02237,
        omch2=0.1200,
        tau=0.0544,
    )

    pars.InitPower.set_params(
        As=2.1e-9,
        ns=0.9649,
        r=r,
    )

    pars.WantTensors = True

    pars.set_for_lmax(lmax)

    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit="muK",
    )

    dl_bb = powers["total"][:, 2]  # lensing + tensor

    ell = np.arange(len(dl_bb))

    cl_bb = np.zeros_like(dl_bb)

    mask = ell >= 2

    cl_bb[mask] = (
        dl_bb[mask]
        * 2.0
        * np.pi
        / (ell[mask] * (ell[mask] + 1))
    )

    return ell, cl_bb
