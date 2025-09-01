/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Linear algebra traits.
 */

#ifndef OPENKALMAN_LINEAR_ALGEBRA_TRAITS_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_TRAITS_HPP

#include <type_traits>

// basic traits

#include "scalar_type_of.hpp"
#include "index_count.hpp"
#include "vector_space_descriptor_of.hpp"
#include "index_dimension_of.hpp"
#include "dynamic_index_count.hpp"
#include "max_tensor_order.hpp"

#include "nested_object_of.hpp"

// constants:

#include "constant_coefficient.hpp"
#include "constant_diagonal_coefficient.hpp"

// special matrices:

#include "triangle_type_of.hpp"
#include "hermitian_adapter_type_of.hpp"

// other:

#include "layout_of.hpp"


#endif
