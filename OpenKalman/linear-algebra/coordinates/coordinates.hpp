/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions relating to \ref coordinate::pattern object.
 *
 * \dir coordinates/descriptors
 * \brief Files defining \ref coordinate::descriptor objects.
 *
 * \dir coordinates/concepts
 * \brief Concepts relating to \ref coordinate::pattern objects.
 *
 * \dir coordinates/functions
 * \brief Files for functions relating to \ref coordinate::pattern objects.
 *
 * \dir coordinates/traits
 * \brief Traits relating to \ref coordinate::pattern objects.
 *
 * \dir coordinates/internal
 * \internal
 * \brief Internal files relating to \ref coordinate::pattern objects.
 *
 * \dir coordinates/tests
 * \internal
 * \brief Tests relating to \ref coordinate::pattern objects.
 *
 * \file
 * \brief Comprehensive header file including all classes and definitions relating to a \ref coordinate::pattern
 */

#ifndef OPENKALMAN_VECTOR_TYPES_HPP
#define OPENKALMAN_VECTOR_TYPES_HPP


/**
 * \internal
 * \brief The namespace for vector-space descriptor code.
 */
namespace OpenKalman::coordinate {}


#include "interfaces/coordinate_descriptor_traits.hpp"


#include "concepts/descriptor.hpp"
#include "concepts/descriptor_tuple.hpp"
#include "concepts/descriptor_range.hpp"
#include "concepts/descriptor_collection.hpp"

#include "concepts/pattern.hpp"
#include "concepts/pattern_tuple.hpp"
#include "concepts/pattern_range.hpp"
#include "concepts/pattern_collection.hpp"

#include "functions/internal/get_descriptor_size.hpp"
#include "functions/get_size.hpp"
#include "traits/size_of.hpp"

#include "functions/internal/get_descriptor_euclidean_size.hpp"
#include "functions/get_euclidean_size.hpp"
#include "traits/euclidean_size_of.hpp"

#include "functions/internal/get_hash_code.hpp"
#include "functions/get_is_euclidean.hpp"

#include "functions/get_component_count.hpp"
#include "traits/component_count_of.hpp"

#include "traits/scalar_type_of.hpp" //

#include "concepts/fixed_pattern.hpp"
#include "concepts/fixed_pattern_tuple.hpp"
#include "concepts/fixed_pattern_collection.hpp"

#include "concepts/dynamic_pattern.hpp"

#include "concepts/euclidean_pattern.hpp"
#include "concepts/euclidean_pattern_tuple.hpp"
#include "concepts/euclidean_pattern_collection.hpp"

#include "functions/internal/get_component_start_indices.hpp"
#include "functions/internal/get_euclidean_component_start_indices.hpp"
#include "functions/internal/get_index_table.hpp"
#include "functions/internal/get_euclidean_index_table.hpp"

#include "functions/internal/get_component_start_indices.hpp"
#include "functions/to_euclidean_element.hpp"
#include "functions/from_euclidean_element.hpp"
#include "functions/get_wrapped_component.hpp"
#include "functions/set_wrapped_component.hpp"

#include "descriptors/Dimensions.hpp"
#include "descriptors/Distance.hpp"
#include "descriptors/Angle.hpp"
#include "descriptors/Inclination.hpp"
#include "descriptors/Polar.hpp"
#include "descriptors/Spherical.hpp"
#include "descriptors/Any.hpp"

#include "functions/comparison-operators.hpp"
#include "concepts/compares_with.hpp"
#include "views/comparison.hpp"


#include "functions/arithmetic-operators.hpp"

// descriptors:

// traits for manipulating static descriptors

#include "traits/internal/uniform_static_vector_space_descriptor_query.hpp"
#include "concepts/uniform_static_vector_space_descriptor.hpp"
#include "traits/uniform_static_vector_space_descriptor_component_of.hpp"
#include "concepts/equivalent_to_uniform_static_vector_space_descriptor_component_of.hpp"

// functions:

#include "functions/internal/is_uniform_component_of.hpp" //
#include "functions/internal/remove_trailing_1D_descriptors.hpp" //
#include "functions/internal/best_vector_space_descriptor.hpp" //
#include "functions/internal/smallest_vector_space_descriptor.hpp" //
#include "functions/internal/largest_vector_space_descriptor.hpp" //
#include "functions/get_slice.hpp"

#include "functions/internal/to_euclidean_vector_space_descriptor_collection.hpp" //


#endif //OPENKALMAN_VECTOR_TYPES_HPP
