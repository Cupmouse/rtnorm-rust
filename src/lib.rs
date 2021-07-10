//  Pseudorandom numbers from a truncated Gaussian distribution.
//
//  This is a port of c++ code.
//
//  This implements an extension of Chopin's algorithm detailed in
//  N. Chopin, "Fast simulation of truncated Gaussian distributions",
//  Stat Comput (2011) 21:275-288
//
//  Original copyright holder:
//  Copyright (C) 2012 Guillaume Dollé, Vincent Mazet
//  (LSIIT, CNRS/Université de Strasbourg)
//  Version 2012-07-04, Contact: vincent.mazet@unistra.fr
//
//  Port copyright holder:
//  Copyright (C) 2021 Nozomu Shimaoka
//
//  07/10/2021:
//  - Ported from c++ version
//  06/07/2012:
//  - first launch of rtnorm.cpp
//
//  Licence: GNU General Public License Version 2
//  This program is free software; you can redistribute it and/or modify it
//  under the terms of the GNU General Public License as published by the
//  Free Software Foundation; either version 2 of the License, or (at your
//  option) any later version. This program is distributed in the hope that
//  it will be useful, but WITHOUT ANY WARRANTY; without even the implied
//  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details. You should have received a
//  copy of the GNU General Public License along with this program; if not,
//  see http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt
//
//  Depends: LibGSL
//  OS: Unix based system
mod table;

use num::ToPrimitive;
use rgsl::{self, error::erf, Rng};

use crate::table::{NCELL, X, YU};

/// Index of the right tail
const N: usize = 4001;

#[cfg(test)]
mod test {
    use rgsl::{Rng, RngType};

    use crate::rtnorm;

    #[test]
    fn generate() -> Result<(), ()> {
        let a = 1.0; // Left bound
        let b = 9.0; // Right bound
        let mu = 2.0; // Mean
        let sigma = 3.0; // Standard deviation
        const K: i64 = 100000; // Number of random variables to generate

        //--- GSL random init ---
        // Read variable environnement
        RngType::env_setup();
        // Rand generator allocation
        // Default algorithm 'twister'
        let mut gen =
            Rng::new(RngType::default()).expect("could not initialize a random number generator");

        //--- generate the random numbers ---
        // println!("# x p(x)");
        // println!("{} {}", a, b);
        // println!("{} {}", mu, sigma);
        for _ in 0..K {
            let (x, _p) = rtnorm(&mut gen, a, b, mu, sigma);
            assert!(a <= x && x <= b);
            // println!("{} {}", s.0, s.1);
        }

        Ok(())
        // GSL rand generator will be deallocated
    }
}

/// Compute y_l from y_k
#[inline]
const fn yl(k: usize) -> f64 {
    // y_l of the leftmost rectangle
    const YL0: f64 = 0.053513975472;
    // y_l of the rightmost rectangle
    const YLN: f64 = 0.000914116389555;

    if k == 0 {
        YL0
    } else if k == N - 1 {
        YLN
    } else if k <= 1953 {
        // TODO: always k > 0 ?
        YU[k - 1]
    } else {
        YU[k + 1]
    }
}

/// Rejection algorithm with a truncated exponential proposal
#[inline]
fn rtexp(gen: &mut Rng, a: f64, b: f64) -> f64 {
    let twoasq = 2.0 * a.powi(2);
    let expab = (-a * (b - a)).exp() - 1.0;
    let mut z;
    let mut e;

    loop {
        z = (1.0 + gen.uniform() * expab).ln();
        e = -(gen.uniform()).ln();
        if twoasq * e > z.powi(2) {
            break;
        }
    }
    return a - z / a;
}

/// Pseudorandom numbers from a truncated Gaussian distribution.
/// The Gaussian has parameters mu and sigma
/// and is truncated on the interval \[a,b\].
/// Returns the random variable x and its probability p(x).
pub fn rtnorm(gen: &mut Rng, mut a: f64, mut b: f64, mu: f64, sigma: f64) -> (f64, f64) {
    // Design variables
    const XMIN: f64 = -2.00443204036; // Left bound
    const XMAX: f64 = 3.48672170399; // Right bound
    const KMIN: i64 = 5; // if kb-ka < kmin then use a rejection algorithm
    const INVH: f64 = 1631.73284006; // = 1/h, h being the minimal interval range
    const I0: i64 = 3271; // = - floor(x(0)/h)
    const ALPHA: f64 = 1.837877066409345; // = log(2*pi)
    const SQ2: f64 = 7.071067811865475e-1; // = 1/sqrt(2)
    const SQPI: f64 = 1.772453850905516; // = sqrt(pi)

    // Scaling
    if mu != 0.0 || sigma != 1.0 {
        a = (a - mu) / sigma;
        b = (b - mu) / sigma;
    }

    // Check if a < b
    assert!(a < b, "B must be greater than A");
    // Check if |a| < |b|
    let r = if a.abs() > b.abs() {
        // Tuple (r,p)
        -rtnorm(gen, -b, -a, 0.0, 1.0).0
    } else if a > XMAX {
        // If a in the right tail (a > xmax), use rejection algorithm with a truncated exponential proposal
        rtexp(gen, a, b)
    } else if a < XMIN {
        // If a in the left tail (a < xmin), use rejection algorithm with a Gaussian proposal
        let mut r;
        loop {
            r = gen.gaussian(1.0);
            if (r >= a) && (r <= b) {
                break;
            }
        }
        r
    } else {
        // In other cases (xmin < a < xmax), use Chopin's algorithm
        // Compute ka
        let ka = NCELL[(I0 + (a * INVH).floor() as i64).to_usize().unwrap()];

        // Compute kb
        let kb = if b >= XMAX {
            N as i64
        } else {
            NCELL[(I0 + (b * INVH).floor() as i64).to_usize().unwrap()]
        };

        // If |b-a| is small, use rejection algorithm with a truncated exponential proposal
        if (kb - ka).abs() < KMIN {
            rtexp(gen, a, b)
        } else {
            loop {
                // Sample integer between ka and kb
                let k = (gen.uniform() * (kb - ka + 1) as f64).floor() as i64 + ka;
                let k = k.to_usize().unwrap();

                if k == N {
                    // Right tail
                    let lbound = X[X.len() - 1];
                    let z = -gen.uniform().ln() / lbound;
                    let e = -gen.uniform().ln();

                    if (z.powi(2) <= 2.0 * e) && (z < b - lbound) {
                        // Accept this proposition, otherwise reject
                        break lbound + z;
                    }
                } else if (k as i64 <= ka + 1) || (k as i64 >= kb - 1 && b < XMAX) {
                    // Two leftmost and rightmost regions
                    let sim = X[k] + (X[k + 1] - X[k]) * gen.uniform();

                    if (sim >= a) && (sim <= b) {
                        // Accept this proposition, otherwise reject
                        let simy = YU[k] * gen.uniform();
                        if (simy < yl(k)) || (sim * sim + 2.0 * simy.ln() + ALPHA) < 0.0 {
                            break sim;
                        }
                    }
                } else {
                    // All the other boxes

                    let u = gen.uniform();
                    let simy = YU[k] * u;
                    let d = X[k + 1] - X[k];
                    if simy < yl(k)
                    // That's what happens most of the time
                    {
                        break X[k] + u * d * YU[k] / yl(k);
                    } else {
                        let sim = X[k] + d * gen.uniform();

                        // Otherwise, check you're below the pdf curve
                        if (sim * sim + 2.0 * simy.ln() + ALPHA) < 0.0 {
                            break sim;
                        }
                    }
                }
            }
        }
    };

    let r = if mu != 0.0 || sigma != 1.0 {
        // Scaling
        r * sigma + mu
    } else {
        r
    };

    // Compute the probability
    let large_z = SQPI * SQ2 * sigma * (erf(b * SQ2) - erf(a * SQ2));
    let p = (-((r - mu) / sigma).powi(2) / 2.0).exp() / large_z;

    return (r, p);
}
