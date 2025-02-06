/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref coordinate_set_traits.
 */

#ifndef OPENKALMAN_COORDINATE_SET_TRAITS_HPP
#define OPENKALMAN_COORDINATE_SET_TRAITS_HPP

#include "basics/internal/tuple_like.hpp"
#include "basics/internal/collection.hpp"

namespace OpenKalman::interface
{
  /**
   * \brief Traits for sets of \ref vector_space_descriptor objects.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct coordinate_set_traits
  {
    static constexpr bool is_specialized = false;


    /**
     * \brief A \ref descriptor::vector_space_descriptor_collection of coordinate descriptors within T.
     */
#ifdef __cpp_concepts
    static constexpr internal::collection auto
#else
    static constexpr auto
#endif
    component_collection(const T& t)
    {
      return std::tuple{};
    }


    /**
     * \brief A \ref internal::collection "collection" mapping a component of T to an \ref value::index "index" within a vector.
     * \details The size must be the same as <code>component_collection(t)</code>.
     * Each component of the resulting collection must map to the corresponding starting index within a vector.
     * \returns A \ref internal::collection "collection" of \ref value::index "index" values
     */
#ifdef __cpp_concepts
    static constexpr internal::collection auto
#else
    static constexpr auto
#endif
    component_start_indices(const T& t)
    {
      return std::array<std::size_t, 0>{};
    }


    /**
     * \brief A \ref internal::collection "collection" mapping a component of T to an \ref value::index "index" within a vector.
     * \details The size must be the same as <code>component_collection(t)</code>.
     * Each component of the resulting collection must map to the corresponding starting index within a vector
     * transformed to Euclidean space for directional statistics.
     * \returns A \ref internal::collection "collection" of \ref value::index "index" values
     */
#ifdef __cpp_concepts
    static constexpr internal::collection auto
#else
    static constexpr auto
#endif
    euclidean_component_start_indices(const T& t)
    {
      return std::array<std::size_t, 0>{};
    }


    /**
     * \brief A \ref internal::collection "collection" mapping each index of an \ref indexible vector
     * to a corresponding \ref value::index "index" within component_collection(t).
     * \returns A \ref internal::collection "collection" of \ref value::index "index" values
     */
#ifdef __cpp_concepts
    static constexpr internal::collection auto
#else
    static constexpr auto
#endif
    index_table(const T& t)
    {
      return std::array<std::size_t, 0>{};
    }


    /**
     * \brief A \ref internal::collection "collection" mapping each index of an \ref indexible vector, transformed to statistical space)
     * to a corresponding \ref value::index "index" within component_collection(t).
     * transformed to Euclidean space for directional statistics.
     * \returns A \ref internal::collection "collection" of \ref value::index "index" values
     */
#ifdef __cpp_concepts
    static constexpr internal::collection auto
#else
    static constexpr auto
#endif
    euclidean_index_table(const T& t)
    {
      return std::array<std::size_t, 0>{};
    }

  };


} // namespace OpenKalman::interface



#endif //OPENKALMAN_COORDINATE_SET_TRAITS_HPP
