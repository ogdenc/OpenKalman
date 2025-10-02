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
 * \file
 * \brief Definition of \ref indexible_object_traits for C++ arrays.
 */

#ifndef OPENKALMAN_INTERFACES_ARRAYS_INDEXIBLE_OBJECT_TRAITS_HPP
#define OPENKALMAN_INTERFACES_ARRAYS_INDEXIBLE_OBJECT_TRAITS_HPP

#include "linear-algebra/interfaces/default/indexible_object_traits.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to standard c++ arrays of any rank.
   */
#ifdef __cpp_concepts
  template<typename T> requires std::is_array_v<T>
  struct indexible_object_traits<T>
#else
  template<typename T>
  struct indexible_object_traits<T, std::enable_if_t<std::is_array_v<T>>>
#endif
  {
  private:

    template<std::size_t rank = 0, std::size_t...es>
    struct mdspan_extents : mdspan_extents<rank + 1, es..., std::extent_v<T, rank>> {};

    template<std::size_t...es>
    struct mdspan_extents<std::rank_v<T>, es...> { using type = stdcompat::extents<std::size_t, es...>; };

    template<typename P>
    static constexpr auto*
    ptr_first_element(P* p)
    {
      if constexpr (std::rank_v<P> > 0)
        return ptr_first_element(p[0]);
      else
        return p;
    }

  public:

    /**
     * \brief Return a std::mdspan as a view to the array.
     */
    static constexpr auto
    get_mdspan = [](auto& t)
    {
      using scalar = std::remove_all_extents_t<std::remove_reference_t<decltype(t)>>;
      using ext = typename mdspan_extents<>::type;
      return stdcompat::mdspan<scalar, ext>{ptr_first_element(t)};
    };

  };

}


#endif
