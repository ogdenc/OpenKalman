/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
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


#include "internal/scalar_traits.hpp"

#include "scalars/complex_number.hpp"
#include "scalars/scalar_type.hpp"
#include "scalars/floating_scalar_type.hpp"
#include "scalars/scalar_constant.hpp"

#include "scalars/get_scalar_constant_value.hpp"
#include "scalars/real_axis_number.hpp"

#include "internal/scalar_constant_operation.hpp"
#include "internal/ScalarConstant.hpp"
#include "internal/scalar-arithmetic.hpp"

#include "internal/make_complex_number.hpp"
#include "internal/are_within_tolerance.hpp"
#include "internal/math_constexpr.hpp"
#include "internal/update_real_part.hpp"

#include "indices/static_index_value.hpp"
#include "indices/dynamic_index_value.hpp"
#include "indices/index_value.hpp"

#include "internal/index_to_scalar_constant.hpp"

#include "internal/collection.hpp"
#include "internal/static_collection_size.hpp"

#endif //OPENKALMAN_VALUES_HPP
