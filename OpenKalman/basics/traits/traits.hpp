/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward declarations for object traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

// basic traits

#include "scalar_type_of.hpp"
#include "indexible.hpp"
#include "index_count.hpp"
#include "vector_space_descriptor_of.hpp"
#include "index_dimension_of.hpp"
#include "dynamic_dimension.hpp"
#include "dynamic_index_count.hpp"
#include "has_dynamic_dimensions.hpp"
#include "dimension_size_of_index_is.hpp"
#include "vector.hpp"
#include "max_tensor_order.hpp"

#include "has_untyped_index.hpp" // Is this necessary?
#include "all_fixed_indices_are_euclidean.hpp" // Is this necessary?

#include "wrappable.hpp"
#include "has_nested_object.hpp"
#include "nested_object_of.hpp"
#include "self_contained.hpp"

// shape-based traits

#include "compatible_with_vector_space_descriptors.hpp"
#include "internal/not_more_fixed_than.hpp"
#include "maybe_same_shape_as.hpp"
#include "same_shape_as.hpp"

#include "one_dimensional.hpp"
#include "square_shaped.hpp"
#include "empty_object.hpp"

// constants:

#include "internal/get_singular_component.hpp"
#include "constant_coefficient.hpp"
#include "constant_matrix.hpp"
#include "zero.hpp"
#include "constant_diagonal_coefficient.hpp"
#include "constant_diagonal_matrix.hpp"
#include "identity_matrix.hpp"

// special matrices:

#include "triangular_matrix.hpp"
#include "triangle_type_of.hpp"
#include "triangular_adapter.hpp"
#include "diagonal_matrix.hpp"
#include "diagonal_adapter.hpp"
#include "hermitian_matrix.hpp"
#include "hermitian_adapter.hpp"
#include "hermitian_adapter_type_of.hpp"

// other:

#include "writable.hpp"
#include "modifiable.hpp" // Is this necessary?

#include "element_gettable.hpp" // deprecated
#include "writable_by_component.hpp"

#include "directly_accessible.hpp"
#include "layout_of.hpp"

#include "object-types.hpp"


#endif //OPENKALMAN_TRAITS_HPP
