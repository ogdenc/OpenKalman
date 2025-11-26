/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2025 Christopher Lee Ogden <ogden@gatech.edu>
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
namespace OpenKalman::values {}

#include "basics/basics.hpp"

#include "constants.hpp"

#include "interface/number_traits.hpp"

#include "concepts/fixed.hpp"
#include "concepts/dynamic.hpp"
#include "concepts/number.hpp"

#include "functions/to_value_type.hpp"
#include "traits/fixed_value_of.hpp"
#include "traits/value_type_of.hpp"
#include "concepts/fixed_value_compares_with.hpp"

#include "concepts/value.hpp"
#include "concepts/complex.hpp"
#include "concepts/integral.hpp"
#include "concepts/index.hpp"
#include "concepts/size.hpp"
#include "concepts/size_compares_with.hpp"
#include "concepts/floating.hpp"
#include "concepts/not_complex.hpp"

#include "functions/operation.hpp"
#include "classes/fixed_value.hpp"
#include "classes/fixed-constants.hpp"

#include "math/real.hpp"
#include "math/imag.hpp"
#include "traits/real_type_of.hpp"
#include "traits/complex_type_of.hpp"

#include "functions/cast_to.hpp"

#include "math/isinf.hpp"
#include "math/isnan.hpp"
#include "math/conj.hpp"
#include "math/signbit.hpp"
#include "math/copysign.hpp"
#include "math/fmod.hpp"
#include "math/sqrt.hpp"
#include "math/hypot.hpp"
#include "math/abs.hpp"
#include "math/exp.hpp"
#include "math/expm1.hpp"
#include "math/sinh.hpp"
#include "math/cosh.hpp"
#include "math/tanh.hpp"
#include "math/sin.hpp"
#include "math/cos.hpp"
#include "math/tan.hpp"
#include "math/log.hpp"
#include "math/log1p.hpp"
#include "math/asinh.hpp"
#include "math/acosh.hpp"
#include "math/atanh.hpp"
#include "math/asin.hpp"
#include "math/acos.hpp"
#include "math/atan.hpp"
#include "math/atan2.hpp"
#include "math/pow.hpp"

#endif
