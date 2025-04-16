/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions relating to collections.
 *
 * \file
 * \brief Inclusion file for collections.
 */

#ifndef OPENKALMAN_COLLECTIONS_HPP
#define OPENKALMAN_COLLECTIONS_HPP

// namespaces

/**
 * \brief Namespace for collections.
 */
namespace OpenKalman::collections {}


/**
 * \brief Namespace for generalized views.
 */
namespace OpenKalman::views {}


#include "basics/basics.hpp"

#include "concepts/sized_random_access_range.hpp"
#include "concepts/tuple_like.hpp"
#include "concepts/collection.hpp"
#include "concepts/index.hpp"
#include "concepts/invocable_on_collection.hpp"

#include "traits/common_tuple_type.hpp"
#include "concepts/uniform_tuple_like.hpp"
#include "traits/common_collection_type.hpp"

#include "traits/size_of.hpp"
#include "functions/get_collection_size.hpp"

#include "functions/get.hpp"

#include "functions/internal/tuple_concatenate.hpp"
#include "functions/internal/tuple_slice.hpp"
#include "functions/internal/tuple_fill.hpp"
#include "functions/internal/tuple_reverse.hpp"
#include "functions/internal/tuple_flatten.hpp"

#include "functions/internal/to_tuple.hpp"

#include "concepts/viewable_collection.hpp"
#include "views/collection_view_interface.hpp"
#include "concepts/collection_view.hpp"

#include "views/all.hpp"
#include "views/single.hpp"
#include "views/reverse.hpp"
#include "views/replicate.hpp"
#include "views/concat.hpp"
#include "views/slice.hpp"

#include "views/iota.hpp"
#include "views/transform.hpp"


#endif //OPENKALMAN_COLLECTIONS_HPP
