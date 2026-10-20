"""Fetch the official Bayesian Machine Scientist (BMS) code and the priors used in this study.

The BMS (Guimera et al., 2020, Sci. Adv. 6:eaav6971) is not redistributed here. This script:
  1. clones https://bitbucket.org/rguimera/machine-scientist into analisis/vendor/machine-scientist
     and checks out the exact commit used in the paper (branch no_degeneracy);
  2. excludes Prior/*.dat from the checkout (their names contain ':' and are invalid on Windows);
  3. extracts the 25 priors used here from git, renames them (' ' and ':' -> '_', content unchanged)
     into analisis/priors/, and verifies their SHA-256.
The BMS code itself is never modified: run-time patches live in analisis/bms_parches.py.

Usage (from the repository root): python analisis/obtener_bms.py      (requires git >= 2.25)
"""
import hashlib, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://bitbucket.org/rguimera/machine-scientist.git'
COMMIT = '9defb7aa2a1382b4a8518f3060e8dd87a20acb99'
VENDOR = ROOT/'analisis'/'vendor'/'machine-scientist'
PRIORS = ROOT/'analisis'/'priors'
PRIOR_SHA256 = {
    'final_prior_param_sq.named_equations.nv2.np10.2016-09-09_18_49_42.600380.dat':
        '959e7727aef287d71ad1eacfbcef9843ee99a4a6df60567604b8f3ea58b9413b',
    'final_prior_param_sq.named_equations.nv2.np10.maxs200.2018-07-20_10_33_53.038930.dat':
        '15008c11acbc9b3f3540653ed53825d59ece93ddd1fe18fa837b24925cd68686',
    'final_prior_param_sq.named_equations.nv2.np11.2016-09-16_10_25_20.360820.dat':
        '10f1c1aec620f0f2433674100c3892d72ab3a54cebb548e39b96654796925a57',
    'final_prior_param_sq.named_equations.nv2.np12.2016-09-16_10_25_20.246564.dat':
        'a05c50f1d3d8d60841b63ecc0e3ce1262bbd267d466a444eafe6acb908370a07',
    'final_prior_param_sq.named_equations.nv2.np13.2016-09-16_10_25_20.246322.dat':
        '24223a1686628bbda7990f338d6dbca7a346441f92316ef7d9dd3b4dcd2a5989',
    'final_prior_param_sq.named_equations.nv2.np14.2016-09-16_10_25_20.244704.dat':
        'c553c389dcf199212d8611c7cc0dcc347bd69f17c5d19d3a424c521b3a7bc9f9',
    'final_prior_param_sq.named_equations.nv2.np15.2016-09-16_10_25_20.903940.dat':
        '2d6cef75e033accab87c0091c0a5906f8e208950d135e1b086d49f3a42709fcb',
    'final_prior_param_sq.named_equations.nv2.np16.2016-09-16_10_25_20.249552.dat':
        '759d244a951fe484018d1510748159cfdf9e149b7a4330415b52f8d740217119',
    'final_prior_param_sq.named_equations.nv2.np17.2016-09-16_10_25_20.219200.dat':
        'cc754ce7e431f62e91be5ada8a1a70322ca7bf677555ad337d4e531e7812b4bf',
    'final_prior_param_sq.named_equations.nv2.np18.2016-09-16_10_25_20.593806.dat':
        '4b83ccdd91227d8636dcdb1aa3b6ebfaf2ce1ed280bf659793eda61f20b3a7bb',
    'final_prior_param_sq.named_equations.nv2.np19.2016-09-16_10_25_20.245986.dat':
        '96a58d3842ccf16cb362d39387effb3cde5cf87e837fc286b380132f15d7e6ec',
    'final_prior_param_sq.named_equations.nv2.np2.2016-09-09_18_49_43.038278.dat':
        '61febb09e9f844b73f31f7523729a80c47dd48cbe9b71c59dce618ffe102b142',
    'final_prior_param_sq.named_equations.nv2.np20.2016-09-16_10_25_20.190011.dat':
        '33959460efd55924fc122313f07c7a0718f190a9114ac5cfa1fa71a5077455a2',
    'final_prior_param_sq.named_equations.nv2.np20.maxs200.2020-06-03_08_05_45.672678.dat':
        'b685261e334a6fbd7236b365428db455f58043c95cf927cb187d7f6c171e879b',
    'final_prior_param_sq.named_equations.nv2.np3.2016-09-09_18_49_42.927679.dat':
        'cf664e9fd7fb93d0b2d62857a4732ab3dcc964b4145b9b382d41705fa2ca0508',
    'final_prior_param_sq.named_equations.nv2.np4.2016-09-09_18_49_43.056910.dat':
        '0ecb4ca2136d86663baf25fcd5e85d8a831b75feeae1d85574af144fa4386d9f',
    'final_prior_param_sq.named_equations.nv2.np5.2016-09-09_18_49_43.097805.dat':
        '91558334231857d1fc131be57839ecbedba63ce1baea78ea54872ab0a7120d41',
    'final_prior_param_sq.named_equations.nv2.np5.maxs200.2018-06-25_10_23_11.337003.dat':
        '75cc161158d29f7d194b689cff485a9491762f412cc984e123e2c051a136f08c',
    'final_prior_param_sq.named_equations.nv2.np6.2016-09-09_18_49_42.600487.dat':
        'b1456eee65d6da6615b2e4f174a2eb22ccf5c1697bd6751243d406dc7abc5dc2',
    'final_prior_param_sq.named_equations.nv2.np7.2016-09-09_18_49_42.827963.dat':
        'ccaee9697e488f971745fcb0f8c14899b53f06b273f4a6f29f5c958a8338def1',
    'final_prior_param_sq.named_equations.nv2.np8.2016-09-09_18_49_42.800618.dat':
        '5f0c46476912eb6ade2c873f4d4686f71a21280b32c25635cc9f9caed199f892',
    'final_prior_param_sq.named_equations.nv2.np9.2016-09-09_18_49_43.025912.dat':
        'd2308034c9b885cfdf2e290858853dacc8ed44d5f39c4e1081d2a8da900d7183',
    'final_prior_param_sq.named_equations.nv3.np3.2017-06-13_08_55_24.082204.dat':
        'd4aa2d74c72b37c0840c6fb55d349728d9a86b4b6bde41442d96136bd41e2778',
    'final_prior_param_sq.named_equations.nv4.np50.maxs200.2020-10-13_11_31_09.949071.dat':
        'f7b1f136e2d7c426ea20448a080a220d37ccb298ed608aeef2cac49c7404f149',
    'final_prior_param_sq.named_equations.nv4.np8.maxs200.2019-12-03_09_30_20.580307.dat':
        'bc05941edbbad747791cc8e4409571ccd7e774c0a6c79f3587958146011c9faf',
}


def git(*args, capture=False):
    r = subprocess.run(['git', '-C', str(VENDOR), *args], check=True, capture_output=capture)
    return r.stdout if capture else None


def main():
    if not (VENDOR/'.git').exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', URL, str(VENDOR)], check=True)
        git('config', 'core.sparseCheckout', 'true')
        (VENDOR/'.git'/'info').mkdir(parents=True, exist_ok=True)
        (VENDOR/'.git'/'info'/'sparse-checkout').write_text('/*\n!/Prior/*.dat\n', encoding='ascii')
        git('checkout', '--detach', COMMIT)
    head = git('rev-parse', 'HEAD', capture=True).decode().strip()
    assert head == COMMIT, f'vendor/machine-scientist is at {head}, expected {COMMIT}'

    names = git('ls-tree', '--name-only', COMMIT, 'Prior/', capture=True).decode('utf-8').splitlines()
    by_new_name = {n.split('/', 1)[1].replace(' ', '_').replace(':', '_'): n for n in names if n.endswith('.dat')}
    PRIORS.mkdir(parents=True, exist_ok=True)
    for new, sha in PRIOR_SHA256.items():
        content = git('show', f'{COMMIT}:{by_new_name[new]}', capture=True)
        assert hashlib.sha256(content).hexdigest() == sha, f'SHA-256 mismatch for {new}'
        (PRIORS/new).write_bytes(content)
    print(f'BMS at commit {COMMIT[:10]} in {VENDOR}')
    print(f'{len(PRIOR_SHA256)} priors written to {PRIORS} (SHA-256 verified)')


if __name__ == '__main__':
    sys.exit(main())
