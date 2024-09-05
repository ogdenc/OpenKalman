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
 * \brief Definitions relating to object properties
 *
 * \file
 * \brief Header file for code relating to object properties
 */

#ifndef OPENKALMAN_PROPERTIES_FUNCTIONS_HPP
#define OPENKALMAN_PROPERTIES_FUNCTIONS_HPP


#include "count_indices.hpp"
#include "get_vector_space_descriptor.hpp"
#include "get_index_dimension_of.hpp"
#include "tensor_order.hpp"
#include "all_vector_space_descriptors.hpp"
#include "vector_space_descriptors_match.hpp"
#include "is_square_shaped.hpp"
#include "is_one_dimensional.hpp"
#include "is_vector.hpp"
#include "get_wrappable.hpp"

#include "internal/index_dimension_scalar_constant.hpp"
#include "internal/raw_data.hpp"
#include "internal/strides.hpp"
#include "internal/has_static_strides.hpp"
#include "internal/truncate_indices.hpp"


#endif //OPENKALMAN_PROPERTIES_FUNCTIONS_HPP
