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
 * \brief Definition for \ref make_dense_object function.
 */

#ifndef OPENKALMAN_MAKE_DENSE_OBJECT_HPP
#define OPENKALMAN_MAKE_DENSE_OBJECT_HPP

namespace OpenKalman
{
  /**
   * \brief Make a default, dense, writable matrix based on a list of Dimensions describing the sizes of each index.
   * \details The result will be uninitialized.
   * \tparam T A dummy matrix or array from the relevant library (size, shape, and layout are ignored)
   * \tparam layout The \ref Layout of the resulting object. If this is Layout::none, it will be the default layout for the library of T.
   * \tparam Scalar The scalar type of the resulting object (by default, it is the same scalar type as T).
   * \param d a tuple of \ref vector_space_descriptor describing dimensions of each index.
   * These can be omitted, in which case the \ref vector_space_descriptor will be derived from T.
   */
#ifdef __cpp_concepts
  template<indexible T, Layout layout = Layout::none, scalar_type Scalar = scalar_type_of_t<T>>
  constexpr writable auto
  make_dense_object(vector_space_descriptor auto&&...d)
    requires (sizeof...(d) > 0 or not has_dynamic_dimensions<T>) and (layout != Layout::stride)
#else
  template<typename T, Layout layout = Layout::none, typename Scalar = scalar_type_of_t<T>, typename...D, std::enable_if_t<
    indexible<T> and scalar_type<Scalar> and (vector_space_descriptor<D> and ...) and
    (sizeof...(D) > 0 or not has_dynamic_dimensions<T>) and (layout != Layout::stride), int> = 0>
  constexpr auto
  make_dense_object(D&&...d)
#endif
  {
    if constexpr (sizeof...(d) == 0 and index_count_v<T> != 0)
    {
      return std::apply(
        [](const auto&...d) {
          return make_dense_object<T, layout, Scalar>(d...);
        }, all_vector_space_descriptors<T>());
    }
    else
    {
      using Traits = interface::library_interface<std::decay_t<T>>;
      return Traits::template make_default<layout, Scalar>(std::forward<decltype(d)>(d)...);
    }
  }



namespace detail
{
  template<typename M, typename Arg, typename...J>
  static void copy_tensor_elements(M& m, const Arg& arg, std::index_sequence<>, J...j)
  {
    set_component(m, get_component(arg, j...), j...);
  }


  template<typename M, typename Arg, std::size_t I, std::size_t...Is, typename...J>
  static void copy_tensor_elements(M& m, const Arg& arg, std::index_sequence<I, Is...>, J...j)
  {
    for (std::size_t i = 0; i < get_index_dimension_of<I>(arg); i++)
      copy_tensor_elements(m, arg, std::index_sequence<Is...> {}, j..., i);
  }


  template<Layout layout, typename Scalar, typename T>
  constexpr auto
  make_default_based_on_arg(const T& t)
  {
    return std::apply(
      [](auto&&...d) {
        return make_dense_object<T, layout, Scalar>(std::forward<decltype(d)>(d)...);
      }, all_vector_space_descriptors(t));
  }

} // namespace detail


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
  make_dense_object(Arg&& arg)
  {
    if constexpr (writable<Arg> and std::is_same_v<Scalar, scalar_type_of_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using M = std::decay_t<decltype(detail::make_default_based_on_arg<layout, Scalar>(arg))>;
      using N = decltype(to_native_matrix<M>(std::declval<Arg&&>()));

      if constexpr (std::is_constructible_v<M, N>)
      {
        return M {to_native_matrix<M>(std::forward<Arg>(arg))};
      }
      else
      {
        auto m {detail::make_default_based_on_arg<layout, Scalar>(arg)};

        if constexpr (std::is_assignable_v<M&, Arg&&>)
        {
          m = std::forward<Arg>;
        }
        else if constexpr (std::is_assignable_v<M&, N>)
        {
          m = to_native_matrix<M>(std::forward<Arg>);
        }
        else
        {
          detail::copy_tensor_elements(m, arg, std::make_index_sequence<index_count_v<Arg>>{});
        }
        return m;
      }
    }
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
  make_dense_object(Arg&& arg)
  {
    return make_dense_object<layout, scalar_type_of_t<Arg>>(std::forward<Arg>(arg));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DENSE_OBJECT_HPP
