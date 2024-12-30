/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Code relating to values (e.g., scalars and indices)
 *
 * \file
 * \brief Header file for code relating to values (e.g., scalars and indices)
 */

#ifndef OPENKALMAN_VALUES_HPP
#define OPENKALMAN_VALUES_HPP

/**
 * \internal
 * \brief The namespace for value-related code.
 */
namespace OpenKalman::value {}


// scalar values

#include "concepts/number.hpp"
#include "concepts/fixed.hpp"
#include "concepts/dynamic.hpp"
#include "concepts/value.hpp"

#include "functions/to_number.hpp"
#include "traits/fixed_number_of.hpp"
#include "traits/number_type_of_t.hpp"
#include "traits/real_type_of_t.hpp"

#include "concepts/complex.hpp"
#include "concepts/integral.hpp"
#include "concepts/index.hpp"
#include "concepts/floating.hpp"
#include "concepts/not_complex.hpp"

#include "classes/operation.hpp"
#include "classes/Fixed.hpp"

#include "functions/cast_to.hpp"
#include "functions/value-arithmetic.hpp"

#include "functions/real.hpp"
#include "functions/imag.hpp"

#include "functions/isinf.hpp"
#include "functions/isnan.hpp"
#include "functions/conj.hpp"
#include "functions/signbit.hpp"
#include "functions/copysign.hpp"
#include "functions/sqrt.hpp"
#include "functions/hypot.hpp"
#include "functions/abs.hpp"
#include "functions/exp.hpp"
#include "functions/expm1.hpp"
#include "functions/sinh.hpp"
#include "functions/cosh.hpp"
#include "functions/tanh.hpp"
#include "functions/sin.hpp"
#include "functions/cos.hpp"
#include "functions/tan.hpp"
#include "functions/log.hpp"
#include "functions/log1p.hpp"
#include "functions/asinh.hpp"
#include "functions/acosh.hpp"
#include "functions/atanh.hpp"
#include "functions/asin.hpp"
#include "functions/acos.hpp"
#include "functions/atan.hpp"
#include "functions/atan2.hpp"
#include "functions/pow.hpp"



#endif //OPENKALMAN_VALUES_HPP
