#!/usr/bin/env python3

import numpy as np
import argparse
import configparser
import fitsio
from itertools import chain
from os.path import exists

def kappa_shift(z):
    '''Hilbert, Hartlap & Schneider (2011)
    https://arxiv.org/abs/1105.3980'''
    return 0.008*z + 0.029*z**2 - 0.0079*z**3 + 0.00065*z**4

def check_overwrite(filename, overwrite):
    if not overwrite and exists(filename):
        raise FileExistsError(filename)

parser = argparse.ArgumentParser(description='Create FLASK input files from 2POINT data.')
parser.add_argument('FITS', metavar='2POINT', help='FITS file in 2POINT format')
parser.add_argument('CONFIG', help='configuration file for FLASK')
parser.add_argument('-w', '--overwrite', action='store_true', help='overwrite existing files')
args = parser.parse_args()

print('read 2POINT file', args.FITS)

fits = fitsio.FITS(args.FITS)

spectra = {}
kernels = {}

for hdu in fits:
    h = hdu.read_header()
    if h.get('2PTDATA'):
        spectra[hdu.get_extname()] = hdu
    if h.get('NZDATA'):
        kernels[hdu.get_extname()] = hdu

print(f'spectra:', ', '.join(spectra.keys()))
print(f'kernels:', ', '.join(kernels.keys()))

fields = {}
cls = {}

for hdu in spectra.values():
    h = hdu.read_header()
    q1, q2 = h['QUANT1'], h['QUANT2']
    k1, k2 = h['KERNEL_1'], h['KERNEL_2']
    b1, b2 = h['N_ZBIN_1'], h['N_ZBIN_2']
    for i1 in range(1, b1+1):
        f1 = fields.setdefault((q1, k1, i1), len(fields))
        for i2 in range(1, b2+1):
            f2 = fields.setdefault((q2, k2, i2), len(fields))
            binsel = hdu.where(f'BIN1 == {i1} && BIN2 == {i2}')
            if len(binsel) > 0:
                cls[f1, f2] = hdu['ANG', 'VALUE'][binsel]

print(f'number of fields: {len(fields)}')

info = np.empty(len(fields),
                dtype=[('field number', int), ('z bin number', int),
                       ('mean', float), ('shift', float), ('field type', int),
                       ('zmin', float), ('zmax', float)])
nzs = {}

for (q, k, i), f in fields.items():
    hdu = kernels[k]
    h = hdu.read_header()

    nz = hdu['Z_MID', f'BIN{i}'][:]
    nz.dtype.names = ['z', 'nz']
    nz['nz'] /= np.trapz(nz['nz'], nz['z'])

    fnum = f+1
    zbin = 1
    if q == 'GPF':
        ftyp = 1
        mean, shift = 0.0, 1.0
    elif q == 'GEF':
        ftyp = 2
        zeff = np.trapz(nz['z']*nz['nz'], nz['z'])
        mean, shift = 0.0, kappa_shift(zeff)
    else:
        raise ValueError(f'unknown field type: {q}')
    zmin, zmax = nz['z'][0], nz['z'][-1]
    info[f] = (fnum, zbin, mean, shift, ftyp, zmin, zmax)

    ngal = h[f'NGAL_{i}']
    nz['nz'] *= ngal
    nzs[f] = nz


print('read FLASK config', args.CONFIG)

config = configparser.ConfigParser(delimiters=':', comment_prefixes='#',
                                   inline_comment_prefixes='#')

with open(args.CONFIG, 'r') as f:
    config.read_file(chain(('[flask]',), f))

fields_info = config.get('flask', 'fields_info')
cl_prefix = config.get('flask', 'cl_prefix')
nz_prefix = config.get('flask', 'selec_z_prefix')

print('write fields info', fields_info)

check_overwrite(fields_info, args.overwrite)
np.savetxt(fields_info, info, fmt='%d %d %f %f %d %f %f', header=', '.join(info.dtype.names))

print('write Cls', cl_prefix)

for (f1, f2), cl in cls.items():
    filename = f'{cl_prefix}f{f1+1}z1f{f2+1}z1.dat'
    check_overwrite(filename, args.overwrite)
    np.savetxt(filename, cl)

print('write Nzs', nz_prefix)

for f, nz in nzs.items():
    filename = f'{nz_prefix}f{f+1}.dat'
    check_overwrite(filename, args.overwrite)
    np.savetxt(filename, nz)
