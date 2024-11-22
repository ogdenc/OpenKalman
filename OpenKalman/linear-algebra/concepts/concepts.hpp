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
 * \brief linear algebra concepts.
 */

#ifndef OPENKALMAN_LINEAR_ALGEBRA_CONCEPTS_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_CONCEPTS_HPP

#include <type_traits>

// basic

#include "indexible.hpp"
#include "dynamic_dimension.hpp"
#include "has_dynamic_dimensions.hpp"
#include "dimension_size_of_index_is.hpp"
#include "vector.hpp"

#include "has_untyped_index.hpp" // Is this necessary?
#include "all_fixed_indices_are_euclidean.hpp" // Is this necessary?

#include "wrappable.hpp"
#include "has_nested_object.hpp"

// shape-based

#include "internal/not_more_fixed_than.hpp"
#include "internal/less_fixed_than.hpp"
#include "internal/maybe_same_shape_as_vector_space_descriptors.hpp"
#include "internal/has_uniform_static_vector_space_descriptors.hpp"

#include "compatible_with_vector_space_descriptor_collection.hpp"
#include "vector_space_descriptors_may_match_with.hpp"
#include "vector_space_descriptors_match_with.hpp"

#include "one_dimensional.hpp"
#include "square_shaped.hpp"
#include "empty_object.hpp"

// constants:

#include "constant_matrix.hpp"
#include "zero.hpp"
#include "constant_diagonal_matrix.hpp"
#include "identity_matrix.hpp"

// special matrices:

#include "triangular_matrix.hpp"
#include "triangular_adapter.hpp"
#include "diagonal_matrix.hpp"
#include "diagonal_adapter.hpp"
#include "hermitian_matrix.hpp"
#include "hermitian_adapter.hpp"

// other:

#include "writable.hpp"

#include "element_gettable.hpp" // deprecated
#include "index_range_for.hpp"
#include "writable_by_component.hpp"

#include "directly_accessible.hpp"

#include "object-types.hpp"


#endif //OPENKALMAN_LINEAR_ALGEBRA_CONCEPTS_HPP
