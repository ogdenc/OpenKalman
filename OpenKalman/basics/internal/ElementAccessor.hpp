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
 * \file
 * \brief Definition of OpenKalman::internal::ElementAccessor
 */

#ifndef OPENKALMAN_ELEMENTACCESSOR_HPP
#define OPENKALMAN_ELEMENTACCESSOR_HPP

#include <functional>

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief An interface to a matrix, to be used for getting and setting the individual components.
   * \tparam Object The object to be reference.
   */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<indexible Object, std::ranges::input_range Indices> requires writable_by_component<Object, Indices> and
    index_value<std::ranges::range_value_t<Indices>>
#else
  template<typename Object, typename Indices>
#endif
  struct ElementAccessor
  {
    using Scalar = scalar_type_of_t<Object>;


#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<typename Arg, std::invocable PreAccess, std::invocable PostSet> requires
      (static_range_size_v<Indices> == dynamic_size or index_count_v<Arg> == dynamic_size or
        static_range_size_v<Indices> >= index_count_v<Arg>) and std::same_as<Scalar, scalar_type_of_t<Arg>>
#else
    template<typename Arg, typename PreAccess, typename PostSet, std::enable_if_t<
      std::is_invocable_v<PreAccess> and std::is_invocable_v<PostSet> and
      index_value<decltype(*std::declval<Indices>().begin())> and
      (static_range_size<Indices>::value == dynamic_size or index_count<Arg>::value == dynamic_size or
        static_range_size<Indices>::value >= index_count<Arg>::value) and
      std::is_same_v<Scalar, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
    ElementAccessor(Arg&& arg, Indices&& indices, PreAccess&& pre_access = []{}, PostSet&& post_set = []{})
      : object {std::forward<Arg>(arg)},
        indices {std::forward<Indices>(indices)},
        before_access {std::forward<decltype(pre_access)>(pre_access)},
        after_set {std::forward<decltype(post_set)>(post_set)} {}


    /// Get an element.
    operator Scalar() const
    {
      before_access();
      return get_component(object, indices);
    }


    /// Set an element.
    void operator=(Scalar s)
    {
      before_access();
      set_component(object, s, indices);
      after_set();
    }

  private:

    Object object;

    Indices indices;

    const std::function<void()> before_access;

    const std::function<void()> after_set;

  };


  // ----------------- //
  //  Deduction guide  //
  // ----------------- //

  template<typename Arg, typename Indices, typename PreAccess, typename PostSet>
  ElementAccessor(Arg&&, Indices&&, PreAccess&&, PostSet&&) -> ElementAccessor<Arg, Indices>;


}

#endif //OPENKALMAN_ELEMENTACCESSOR_HPP
