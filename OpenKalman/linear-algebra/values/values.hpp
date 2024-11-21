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

// indices

#include "concepts/static_index.hpp"
#include "concepts/dynamic_index.hpp"
#include "concepts/index.hpp"

// scalar values

#include "traits/number_traits.hpp"

#include "concepts/complex_number.hpp"
#include "concepts/number.hpp"
#include "concepts/floating_number.hpp"

#include "concepts/static_scalar.hpp"
#include "concepts/dynamic_scalar.hpp"
#include "concepts/scalar.hpp"

#include "functions/to_number.hpp"
#include "concepts/real_scalar.hpp"

#include "internal-classes/static_scalar_operation.hpp"
#include "internal-classes/StaticScalar.hpp"

#include "functions/internal/scalar-arithmetic.hpp"
#include "functions/internal/make_complex_number.hpp"
#include "functions/internal/are_within_tolerance.hpp"
#include "functions/internal/math_constexpr.hpp"
#include "functions/internal/update_real_part.hpp"
#include "functions/internal/index_to_scalar_constant.hpp"


#endif //OPENKALMAN_VALUES_HPP
