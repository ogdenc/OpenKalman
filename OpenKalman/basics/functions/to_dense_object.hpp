/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref to_dense_object function.
 */

#ifndef OPENKALMAN_TO_DENSE_OBJECT_HPP
#define OPENKALMAN_TO_DENSE_OBJECT_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, Layout layout, typename Scalar, typename Arg>
    constexpr auto
    make_default_based_on_arg(const Arg& arg)
    {
      return std::apply(
        [](auto&&...d) {
          return make_dense_object<T, layout, Scalar>(std::forward<decltype(d)>(d)...);
        }, all_vector_space_descriptors(arg));
    }

  } // namespace detail


  /**
   * \brief Convert the argument to a dense, writable matrix of a particular scalar type.
   * \tparam T A dummy matrix or array from the relevant library (size, shape, and layout are ignored)
   * \tparam layout The \ref Layout of the resulting object. If this is Layout::none, the interface will decide the layout.
   * \tparam Scalar The Scalar type of the new matrix, if different than that of Arg
   * \param arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type Scalar = scalar_type_of_t<T>, indexible Arg> requires
    (layout != Layout::stride)
  constexpr writable decltype(auto)
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename Arg, std::enable_if_t<
    indexible<T> and (layout != Layout::stride) and scalar_type<Scalar> and indexible<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  to_dense_object(Arg&& arg)
  {
    if constexpr (writable<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (writable<decltype(to_native_matrix<T>(std::declval<Arg&&>()))>)
    {
      return to_native_matrix<T>(std::forward<Arg>(arg));
    }
    else
    {
      auto m {detail::make_default_based_on_arg<T, layout, Scalar>(arg)};
      assign(m, std::forward<Arg>(arg));
      return m;
    }
  }


/**
 * \brief Convert the argument to a dense, writable matrix of a particular scalar type.
 * \tparam layout The \ref Layout of the resulting object. If this is Layout::none, the interface will decide the layout.
 * \tparam Scalar The Scalar type of the new matrix, if different than that of Arg
 * \param arg The object from which the new matrix is based
 */
#ifdef __cpp_concepts
  template<Layout layout, scalar_type Scalar, indexible Arg> requires (layout != Layout::stride)
  constexpr writable decltype(auto)
#else
  template<Layout layout, typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar> and indexible<Arg> and
    (layout != Layout::stride), int> = 0>
  constexpr decltype(auto)
#endif
  to_dense_object(Arg&& arg)
  {
    if constexpr (writable<Arg> and std::is_same_v<Scalar, scalar_type_of_t<Arg>>)
      return std::forward<Arg>(arg);
    else
      return to_dense_object<std::decay_t<Arg>, layout, Scalar>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Convert the argument to a dense, writable matrix with the same scalar type as the argument.
   * \tparam layout The \ref Layout of the resulting object (optional). If this is omitted or Layout::none,
   * the interface will decide the layout.
   * \param arg The object from which the new matrix is based
   */
#ifdef __cpp_concepts
  template<Layout layout = Layout::none, indexible Arg> requires (layout != Layout::stride)
  constexpr writable decltype(auto)
#else
  template<Layout layout = Layout::none, typename Arg, std::enable_if_t<indexible<Arg> and (layout != Layout::stride), int> = 0>
  constexpr decltype(auto)
#endif
  to_dense_object(Arg&& arg)
  {
    if constexpr (writable<Arg>)
      return std::forward<Arg>(arg);
    else
      return to_dense_object<std::decay_t<Arg>, layout, scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_DENSE_OBJECT_HPP
