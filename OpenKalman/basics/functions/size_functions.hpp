/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general functions relating to object size or dimension.
 */

#ifndef OPENKALMAN_SIZE_FUNCTIONS_HPP
#define OPENKALMAN_SIZE_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // --------------------- //
  //  get_tensor_order_of  //
  // --------------------- //

  namespace detail
  {
    template<std::size_t...I, typename T>
    constexpr auto get_tensor_order_of_impl(std::index_sequence<I...>, const T& t)
    {
      return ((IndexTraits<T, I>::dimension_at_runtime(t) == 1 ? 0 : 1) + ... + 0);
    }
  }


  /**
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto get_tensor_order_of(const T& t)
  {
    if constexpr (not has_dynamic_dimensions<T>)
      return tensor_order_of_v<T>;
    else
      return detail::get_tensor_order_of_impl(std::make_index_sequence<max_indices_of_v<T>> {}, t);
  }


  // -------------------------- //
  //   get_index_dimension_of   //
  // -------------------------- //

  /**
   * \brief Get the runtime dimensions of index N of \ref indexible T
   */
#ifdef __cpp_concepts
  template<std::size_t N, indexible T>
#else
  template<std::size_t N, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr std::size_t
  get_index_dimension_of(const T& t)
  {
    constexpr auto dim = index_dimension_of_v<T, N>;
    if constexpr (dim == dynamic_size) return IndexTraits<T, N>::dimension_at_runtime(t);
    else return dim;
  }


  // ------------------- //
  //  get_dimensions_of  //
  // ------------------- //

  /**
   * \brief Get the index descriptor of object Arg for index N.
   */
#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible Arg> requires (N < max_indices_of_v<Arg>) and
    (euclidean_index_descriptor<coefficient_types_of_t<Arg, N>> or
      requires(const Arg& arg) { interface::CoordinateSystemTraits<Arg, N>::coordinate_system_types_at_runtime(arg); })
#else
  template<std::size_t N = 0, typename Arg, std::enable_if_t<indexible<Arg> and N < max_indices_of<Arg>::value, int> = 0>
#endif
  constexpr auto get_dimensions_of(const Arg& arg)
  {
    using T = coefficient_types_of_t<Arg, N>;
    if constexpr (euclidean_index_descriptor<T>)
    {
      if constexpr (dynamic_dimension<Arg, N>)
        return Dimensions{interface::IndexTraits<std::decay_t<Arg>, N>::dimension_at_runtime(arg)};
      else
        return Dimensions<index_dimension_of_v<Arg, N>> {};
    }
    else
    {
      if constexpr (dynamic_dimension<Arg, N>)
        return interface::CoordinateSystemTraits<Arg, N>::coordinate_system_types_at_runtime(arg);
      else
        return coefficient_types_of_t<Arg, N> {};
    }
  }


  // ----------------------- //
  //  get_all_dimensions_of  //
  // ----------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(const T& t, std::index_sequence<I...>)
    {
      return std::tuple {get_dimensions_of<I>(t)...};
    }


    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>)
    {
      return std::tuple {coefficient_types_of_t<T, I> {}...};
    }
  }


  /**
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto) get_all_dimensions_of(const T& t)
  {
    return detail::get_all_dimensions_of_impl(t, std::make_index_sequence<max_indices_of_v<T>> {});
  }


  /**
   * \overload
   * \brief Return a tuple of \ref index_descriptor objects defining the dimensions of T.
   * \details This overload is only enabled if all dimensions of T are known at compile time.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T> requires (not has_dynamic_dimensions<T>)
#else
  template<typename T, std::enable_if_t<indexible<T> and not has_dynamic_dimensions<T>, int> = 0>
#endif
  constexpr auto get_all_dimensions_of()
  {
    return detail::get_all_dimensions_of_impl<T>(std::make_index_sequence<max_indices_of_v<T>> {});
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SIZE_FUNCTIONS_HPP
