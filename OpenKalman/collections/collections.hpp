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
namespace OpenKalman::collections::views {}


namespace OpenKalman
{
  /**
   * \brief Namespace alias for generalized views.
   */
  namespace views = collections::views;
}

#include "values/values.hpp"

#include "concepts/sized.hpp"
#include "functions/get_size.hpp"
#include "traits/size_of.hpp"

#include "concepts/sized_random_access_range.hpp"
#include "concepts/gettable.hpp"
#include "concepts/uniformly_gettable.hpp"
#include "concepts/tuple_like.hpp"
#include "concepts/collection.hpp"
#include "concepts/index.hpp"
#include "concepts/invocable_on_collection.hpp"

#include "concepts/settable.hpp"
#include "concepts/uniformly_settable.hpp"
#include "concepts/output_collection.hpp"

#include "traits/collection_element.hpp"
#include "traits/common_collection_type.hpp"
#include "concepts/viewable_tuple_like.hpp"
#include "functions/get.hpp"

#include "functions/lexicographical_compare_three_way.hpp"

#include "concepts/viewable_collection.hpp"
#include "concepts/collection_view.hpp"

#include "functions/internal/tuple_reverse.hpp"
#include "functions/internal/tuple_flatten.hpp"
#include "functions/internal/tuple_like_to_tuple.hpp"
#include "functions/apply.hpp"

#include "views/from_tuple_like_range.hpp"
#include "views/from_tuple_like.hpp"
#include "views/from_range.hpp"
#include "views/all.hpp"
#include "views/replicate.hpp"
#include "views/generate.hpp"
#include "views/slice.hpp"
#include "views/iota.hpp"
#include "views/repeat.hpp"
#include "views/concat.hpp"


#endif
