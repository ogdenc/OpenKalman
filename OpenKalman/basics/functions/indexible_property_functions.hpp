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
 * \brief Overloaded general functions relating to object size, dimension, or other index properties.
 */

#ifndef OPENKALMAN_INDEXIBLE_PROPERTY_FUNCTIONS_HPP
#define OPENKALMAN_INDEXIBLE_PROPERTY_FUNCTIONS_HPP

#include<optional>


namespace OpenKalman
{
  using namespace interface;

  // ---------------------- //
  //  get_index_descriptor  //
  // ---------------------- //

  /**
   * \brief Get the index descriptor of object Arg for index N.
   */
#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible Arg> requires (N < max_indices_of_v<Arg>)
  constexpr index_descriptor auto get_index_descriptor(const Arg& arg)
#else
  template<std::size_t N = 0, typename Arg, std::enable_if_t<indexible<Arg> and N < max_indices_of<Arg>::value, int> = 0>
  constexpr auto get_index_descriptor(const Arg& arg)
#endif
  {
    return interface::IndexTraits<Arg>::template get_index_descriptor<N>(arg);
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
    return get_dimension_size_of(get_index_descriptor<N>(t));
  }


  // --------------------- //
  //  get_tensor_order_of  //
  // --------------------- //

  namespace detail
  {
    template<typename T>
    constexpr std::size_t get_tensor_order_of_impl(std::index_sequence<>, const T& t) { return 0; }

    template<std::size_t I, std::size_t...Is, typename T>
    constexpr std::size_t get_tensor_order_of_impl(std::index_sequence<I, Is...>, const T& t)
    {
      std::size_t dim = get_index_dimension_of<I>(t);
      if (dim == 0) return 0;
      else if (dim == 1) return get_tensor_order_of_impl(std::index_sequence<Is...> {}, t);
      else return 1 + get_tensor_order_of_impl(std::index_sequence<Is...> {}, t);
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
  constexpr std::size_t get_tensor_order_of(const T& t)
  {
    if constexpr (not has_dynamic_dimensions<T>) return max_tensor_order_of_v<T>;
    else return detail::get_tensor_order_of_impl(std::make_index_sequence<max_indices_of_v<T>> {}, t);
  }


  // ----------------------- //
  //  get_all_dimensions_of  //
  // ----------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(const T& t, std::index_sequence<I...>)
    {
      return std::tuple {get_index_descriptor<I>(t)...};
    }


    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>)
    {
      return std::tuple {index_descriptor_of_t<T, I> {}...};
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


  // ----------------------------- //
  //  get_index_descriptors_match  //
  // ----------------------------- //

  namespace detail
  {
    template<std::size_t...Is>
    constexpr bool get_index_descriptors_match_impl(std::index_sequence<Is...>) { return true; }

    template<std::size_t...Is, typename T, typename...Ts>
    constexpr bool get_index_descriptors_match_impl(std::index_sequence<Is...>, const T& t, const Ts&...ts)
    {
      return ([](auto I_const, const T& t, const Ts&...ts){
        constexpr std::size_t I = decltype(I_const)::value;
        return ((get_index_descriptor<I>(t) == get_index_descriptor<I>(ts)) and ...);
      }(std::integral_constant<std::size_t, Is>{}, t, ts...) and ...);
    }
  }


  /**
   * \brief Return true if every \ref index_descriptor of a set of objects match.
   * \tparam Ts A set of tensors or matrices
   */
#ifdef __cpp_concepts
  template<indexible...Ts>
#else
  template<typename...Ts, std::enable_if_t<(indexible<Ts> and ...), int> = 0>
#endif
  constexpr bool get_index_descriptors_match(const Ts&...ts)
  {
    return detail::get_index_descriptors_match_impl(std::make_index_sequence<std::max({max_indices_of_v<Ts>...})> {}, ts...);
  }


  // --------------- //
  //  get_is_square  //
  // --------------- //

  namespace detail
  {
    template<std::size_t I, std::size_t...Is, typename T>
    constexpr auto get_best_square_index_descriptor(std::index_sequence<I, Is...>, const T& t)
    {
      if constexpr (not dynamic_dimension<T, I> or sizeof...(Is) == 0) return get_index_descriptor<I>(t);
      else return get_best_square_index_descriptor(std::index_sequence<Is...>{}, t);
    }


    template<std::size_t I, std::size_t...Is, typename T>
    constexpr auto get_is_square_impl(std::index_sequence<I, Is...>, const T& t)
    {
      auto dim_I = get_index_descriptor<I>(t);
      if (((get_dimension_size_of(dim_I) != 0) and ... and (dim_I == get_index_descriptor<Is>(t))))
        return std::optional {dim_I};
      else return std::optional<decltype(dim_I)> {};
    }
  }


  /**
   * \brief Return true if T is a \ref square_matrix at runtime.
   * \tparam T A tensor or matrix
   * \return a \ref std::optional which includes the index descriptor if T is square.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto get_is_square(const T& t)
  {
    if constexpr (square_matrix<T>)
      return std::optional {detail::get_best_square_index_descriptor(std::make_index_sequence<max_indices_of_v<T>>{}, t)};
    else if constexpr (not square_matrix<T, Likelihood::maybe>)
      return std::optional<std::size_t> {};
    else if constexpr (max_indices_of_v<T> == 1 and dimension_size_of_index_is<T, 0, 1, Likelihood::maybe>)
    {
      auto d = get_index_descriptor<0>(t);
      if (get_dimension_size_of(d) == 1) return std::optional {d};
      else return std::optional<decltype(d)> {};
    }
    else return detail::get_is_square_impl(std::make_index_sequence<max_indices_of_v<T>>{}, t);
  }


  // ------------------- //
  //  get_is_one_by_one  //
  // ------------------- //

  namespace detail
  {
    template<std::size_t...Is, typename T>
    constexpr bool get_is_one_by_one_impl(std::index_sequence<Is...>, const T& t)
    {
      return (... and (get_index_dimension_of<Is>(t) == 1));
    }
  }


  /**
   * \brief Return true if T is a \ref square_matrix at runtime.
   * \tparam T A tensor or matrix
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr bool get_is_one_by_one(const T& t)
  {
    if constexpr (one_by_one_matrix<T>) return true;
    else if constexpr (not one_by_one_matrix<T, Likelihood::maybe>) return false;
    else return detail::get_is_one_by_one_impl(std::make_index_sequence<max_indices_of_v<T>>{}, t);
  }


  // --------------- //
  //  nested_matrix  //
  // --------------- //

  /**
   * \brief Retrieve a nested matrix of Arg, if it exists.
   * \tparam i Index of the nested matrix (0 for the 1st, 1 for the 2nd, etc.).
   * \tparam Arg A wrapper that has at least one nested matrix.
   * \internal \sa interface::Dependencies::get_nested_matrix
   */
#ifdef __cpp_concepts
  template<std::size_t i = 0, typename Arg> requires
    (i < std::tuple_size_v<typename Dependencies<std::decay_t<Arg>>::type>) and
    requires(Arg&& arg) { Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg)); }
#else
  template<std::size_t i = 0, typename Arg,
    std::enable_if_t<(i < std::tuple_size<typename Dependencies<std::decay_t<Arg>>::type>::value), int> = 0,
    typename = std::void_t<decltype(Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::declval<Arg&&>()))>>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg)
  {
      return Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg));
  }


  // --------------- //
  //  get_wrappable  //
  // --------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool get_wrappable_impl(const T& t, std::index_sequence<I...>)
    {
      return (get_index_descriptor_is_euclidean(get_index_descriptor<I + 1>(t)) and ...);
    }
  }


  /**
   * \brief Determine whether T is wrappable (i.e., all its dimensions other than potentially 0 are euclidean).
   * \tparam T A matrix or array
   * \sa wrappable
   */
#ifdef __cpp_concepts
  template<indexible T> requires (max_indices_of_v<T> >= 1)
#else
  template<typename T, std::enable_if_t<indexible<T> and (max_indices_of_v<T> >= 1), int> = 0>
#endif
  constexpr bool get_wrappable(const T& t)
  {
    return detail::get_wrappable_impl(t, std::make_index_sequence<max_indices_of_v<T> - 1> {});
  }


  namespace internal
  {
    // ------------------------------------ //
    //  index_dimension_scalar_constant_of  //
    // ------------------------------------ //

    /**
     * \internal
     * \brief Returns a scalar constant reflecting the size of an index for a tensor or matrix.
     * \details The return value is a known or unknown \ref scalar_constant of the same scalar type as T.
     * \tparam N The index
     * \tparam T The matrix, expression, or array
     * \internal \sa interface::IndexTraits
     */
#ifdef __cpp_concepts
    template<std::size_t N, indexible T>
#else
    template<std::size_t N, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
    constexpr auto index_dimension_scalar_constant_of(const T& t)
    {
      using Scalar = scalar_type_of_t<T>;
      if constexpr (dynamic_dimension<T, N>)
        return static_cast<Scalar>(get_index_dimension_of<N>(t));
      else
        return ScalarConstant<Likelihood::definitely, Scalar, index_dimension_of_v<T, N>>{};
    }

  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_INDEXIBLE_PROPERTY_FUNCTIONS_HPP
