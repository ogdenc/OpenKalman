/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions relating to \ref vector_space_descriptor.
 *
 * \dir vector-space-descriptor/details
 * \brief Files implementing details regarding \ref vector_space_descriptor objects.
 *
 * \dir vector-space-descriptor/tests
 * \brief Test files for coefficient types.
 *
 * \file
 * \brief Comprehensive header file including all index-type-related classes and definitions
 */

#ifndef OPENKALMAN_VECTOR_TYPES_HPP
#define OPENKALMAN_VECTOR_TYPES_HPP

#include "details/vector-space-descriptor-interface-traits.hpp"
#include "details/vector-space-descriptor-forward-declarations.hpp"
#include "basics/functions/vector-space_descriptor-forward-functions.hpp"

#include "details/integral-interface-traits.hpp"

#include "Dimensions.hpp"
#include "Distance.hpp"
#include "Angle.hpp"
#include "Inclination.hpp"
#include "Polar.hpp"
#include "Spherical.hpp"

#include "TypedIndex.hpp"

#include "details/AnyAtomicVectorSpaceDescriptor.hpp"
#include "DynamicTypedIndex.hpp"

#include "details/vector-space-descriptor-traits.hpp"
#include "basics/functions/vector-space_descriptor-functions.hpp"

#endif //OPENKALMAN_VECTOR_TYPES_HPP
