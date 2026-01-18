/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions relating to \ref patterns::pattern object.
 *
 * \dir descriptors
 * \brief Files defining \ref patterns::descriptor objects.
 *
 * \dir concepts
 * \brief Concepts relating to \ref patterns::pattern objects.
 *
 * \dir functions
 * \brief Files for functions relating to \ref patterns::pattern objects.
 *
 * \dir traits
 * \brief Traits relating to \ref patterns::pattern objects.
 *
 * \dir internal
 * \internal
 * \brief Internal files relating to \ref patterns::pattern objects.
 *
 * \dir tests
 * \internal
 * \brief Tests relating to \ref patterns::pattern objects.
 *
 * \file
 * \brief Comprehensive header file including all classes and definitions relating to a \ref patterns::pattern
 */

#ifndef OPENKALMAN_PATTERNS_HPP
#define OPENKALMAN_PATTERNS_HPP

/**
 * \brief The namespace for features relating to \ref patterns::pattern object.
 */
namespace OpenKalman::patterns {}


/**
 * \brief The namespace for views for \ref patterns::pattern object.
 */
namespace OpenKalman::patterns::views {}


#include "collections/collections.hpp"

#include "interfaces/pattern_descriptor_traits.hpp"

#include "concepts/descriptor.hpp"
#include "concepts/descriptor_collection.hpp"

#include "concepts/pattern.hpp"
#include "concepts/sized_pattern.hpp"

#include "functions/get_dimension.hpp"
#include "traits/dimension_of.hpp"

#include "functions/get_stat_dimension.hpp"
#include "traits/stat_dimension_of.hpp"

#include "functions/get_is_euclidean.hpp"

#include "concepts/fixed_pattern.hpp"
#include "concepts/dynamic_pattern.hpp"
#include "concepts/euclidean_pattern.hpp"

#include "functions/to_stat_space.hpp"
#include "functions/from_stat_space.hpp"
#include "functions/wrap.hpp"

// descriptors and their operators

#include "descriptors/Any.hpp"
#include "descriptors/Dimensions.hpp"
#include "descriptors/Distance.hpp"
#include "descriptors/Angle.hpp"
#include "descriptors/Inclination.hpp"
#include "descriptors/Polar.hpp"
#include "descriptors/Spherical.hpp"

// comparisons

#include "traits/common_descriptor_type.hpp"
#include "functions/compare_three_way.hpp"
#include "functions/compare.hpp"
#include "concepts/compares_with.hpp"

// uniform types

#include "traits/uniform_pattern_type.hpp"
#include "concepts/uniform_pattern.hpp"
#include "functions/get_uniform_pattern_component.hpp"
#include "functions/is_uniform_pattern_component_of.hpp"

// pattern_collections:

#include "concepts/pattern_collection.hpp"
#include "concepts/fixed_pattern_collection.hpp"
#include "concepts/euclidean_pattern_collection.hpp"

#include "functions/compare_pattern_collections.hpp"
#include "concepts/collection_compares_with.hpp"

#include "functions/to_extents.hpp"

#include "functions/get_pattern.hpp"
#include "traits/pattern_collection_element.hpp"

#include "concepts/collection_patterns_compare_with_dimension.hpp"
#include "functions/compare_collection_patterns_with_dimension.hpp"

#include "concepts/collection_patterns_have_same_dimension.hpp"
#include "functions/get_common_pattern_collection_dimension.hpp"

// views

#include "views/concat.hpp"
#include "views/replicate.hpp"
#include "views/dimensions.hpp"
#include "views/diagonal_of.hpp"
#include "views/to_diagonal.hpp"
#include "views/transpose.hpp"

#endif
