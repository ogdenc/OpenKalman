/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref make_self_contained.
 */

#ifndef OPENKALMAN_MAKE_SELF_CONTAINED_HPP
#define OPENKALMAN_MAKE_SELF_CONTAINED_HPP

namespace OpenKalman
{
  /**
   * \brief Convert to a self-contained version of Arg that can be returned in a function.
   * \details If any types Ts are included, Arg will not be converted to a self-contained version if every Ts is either
   * an lvalue reference or has a nested matrix that is an lvalue reference. This is to allow a function, taking Ts...
   * as lvalue-reference inputs or as rvalue-reference inputs that nest lvalue-references to other matrices, to avoid
   * unnecessary conversion because the referenced objects are accessible outside the scope of the function and do not
   * result in dangling references.
   * The following example adds two matrices arg1 and arg2 together and returns a self-contained matrix, unless
   * <em>both</em> Arg1 and Arg2 are lvalue references or their nested matrices are lvalue references, in which case
   * the result of the addition is returned without eager evaluation:
   * \code
   *   template<typename Arg1, typename Arg2>
   *   auto add(Arg1&& arg1, Arg2&& arg2)
   *   {
   *     return make_self_contained<Arg1, Arg2>(arg1 + arg2);
   *   }
   * \endcode
   * \tparam Ts Generally, these will be forwarding-reference arguments to the directly enclosing function. If all of
   * Ts... are lvalue references, Arg is returned without modification (i.e., without any potential eager evaluation).
   * \tparam Arg The potentially non-self-contained argument to be converted
   * \return A self-contained version of Arg (if it is not already self-contained)
   * \todo Return a new class that internalizes any external dependencies
   * \internal \sa interface::indexible_object_traits
   */
#ifdef __cpp_concepts
  template<indexible...Ts, indexible Arg>
  constexpr /*self_contained<Ts...>*/ decltype(auto)
#else
  template<typename...Ts, typename Arg, std::enable_if_t<(indexible<Ts> and ...) and indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr (self_contained<Arg, Ts...>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (std::is_lvalue_reference_v<Arg> and self_contained<std::decay_t<Arg>> and
      std::is_copy_constructible_v<std::decay_t<Arg>>)
    {
      // If it is not self-contained because it is an lvalue reference, simply return a copy.
      return std::decay_t<Arg> {arg};
    }
    else if constexpr (interface::convert_to_self_contained_defined_for<Arg&&>)
    {
      return interface::indexible_object_traits<std::decay_t<Arg>>::convert_to_self_contained(std::forward<Arg>(arg));
    }
    else
    {
      // Ensure that copying occurs if Arg is a writable lvalue reference
      auto ret {make_dense_object(std::forward<Arg>(arg))};
      return ret;
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_SELF_CONTAINED_HPP
