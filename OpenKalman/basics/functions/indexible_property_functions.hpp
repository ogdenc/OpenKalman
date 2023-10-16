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
  // ----------------- //
  //  get_index_count  //
  // ----------------- //

  /**
   * \brief Get the number of indices available to address the components of an \ref indexible object.
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr index_value auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  get_index_count(const T& t)
  {
    return interface::indexible_object_traits<T>::get_index_count(t);
  }


  // ------------------ //
  //  get_vector_space_descriptor  //
  // ------------------ //

  /**
   * \brief Get the \ref vector_space_descriptor object for index N of \ref indexible object Arg.
   */
#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible T>
  constexpr vector_space_descriptor auto
#else
  template<std::size_t N = 0, typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  get_vector_space_descriptor(const T& t)
  {
    if constexpr (N < index_count_v<T>)
      return interface::indexible_object_traits<T>::get_vector_space_descriptor(t, std::integral_constant<std::size_t, N>{});
    else
      return Dimensions<1>{};
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible T, index_value N = std::integral_constant<std::size_t, 0>>
  constexpr vector_space_descriptor auto
#else
  template<typename T, typename N = std::integral_constant<std::size_t, 0>, std::enable_if_t<indexible<T> and index_value<N>, int> = 0>
  constexpr auto
#endif
  get_vector_space_descriptor(const T& t, N n = N{})
  {
    return interface::indexible_object_traits<T>::get_vector_space_descriptor(t, n);
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
    return get_dimension_size_of(get_vector_space_descriptor<N>(t));
  }


  /**
   * \overload
   */
#ifdef __cpp_concepts
  template<indexible T, index_value N = std::integral_constant<std::size_t, 0>>
#else
  template<typename T, typename N = std::integral_constant<std::size_t, 0>, std::enable_if_t<indexible<T> and index_value<N>, int> = 0>
#endif
  constexpr std::size_t
  get_index_dimension_of(const T& t, N n = N{})
  {
    return get_dimension_size_of(get_vector_space_descriptor(t, n));
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
   * \brief Return a tuple of \ref vector_space_descriptor defining the dimensions of T.
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
    else return detail::get_tensor_order_of_impl(std::make_index_sequence<index_count_v<T>> {}, t);
  }


  // ----------------------- //
  //  get_all_dimensions_of  //
  // ----------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(const T& t, std::index_sequence<I...>)
    {
      return std::tuple {get_vector_space_descriptor<I>(t)...};
    }


    template<typename T, std::size_t...I>
    constexpr auto get_all_dimensions_of_impl(std::index_sequence<I...>)
    {
      return std::tuple {vector_space_descriptor_of_t<T, I> {}...};
    }
  }


  /**
   * \brief Return a tuple of \ref vector_space_descriptor defining the dimensions of T.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr decltype(auto) get_all_dimensions_of(const T& t)
  {
    return detail::get_all_dimensions_of_impl(t, std::make_index_sequence<index_count_v<T>> {});
  }


  /**
   * \overload
   * \brief Return a tuple of \ref vector_space_descriptor defining the dimensions of T.
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
    return detail::get_all_dimensions_of_impl<T>(std::make_index_sequence<index_count_v<T>> {});
  }


  // ----------------------------- //
  //  get_vector_space_descriptor_match  //
  // ----------------------------- //

  namespace detail
  {
    template<std::size_t...Is>
    constexpr bool get_vector_space_descriptor_match_impl(std::index_sequence<Is...>) { return true; }

    template<std::size_t...Is, typename T, typename...Ts>
    constexpr bool get_vector_space_descriptor_match_impl(std::index_sequence<Is...>, const T& t, const Ts&...ts)
    {
      return ([](auto I_const, const T& t, const Ts&...ts){
        constexpr std::size_t I = decltype(I_const)::value;
        return ((get_vector_space_descriptor<I>(t) == get_vector_space_descriptor<I>(ts)) and ...);
      }(std::integral_constant<std::size_t, Is>{}, t, ts...) and ...);
    }
  }


  /**
   * \brief Return true if every set of \ref vector_space_descriptor of a set of objects match.
   * \tparam Ts A set of tensors or matrices
   */
#ifdef __cpp_concepts
  template<indexible...Ts>
#else
  template<typename...Ts, std::enable_if_t<(indexible<Ts> and ...), int> = 0>
#endif
  constexpr bool get_vector_space_descriptor_match(const Ts&...ts)
  {
    return detail::get_vector_space_descriptor_match_impl(std::make_index_sequence<std::max({index_count_v<Ts>...})> {}, ts...);
  }


  // --------------- //
  //  get_is_square  //
  // --------------- //

  namespace detail
  {
    template<typename T, std::size_t i = 0>
    constexpr auto get_best_square_index()
    {
      if constexpr (i + 1 >= index_count_v<T>) return i;
      else if constexpr (not dynamic_dimension<T, i>) return i;
      else return get_best_square_index<T, i + 1>();
    }


    template<std::size_t...Is, typename T>
    constexpr auto get_is_square_impl(std::index_sequence<Is...>, const T& t)
    {
      constexpr auto bestI = get_best_square_index<T>();
      auto dim_bestI = get_vector_space_descriptor<bestI>(t);
      if ((... and (Is == bestI or get_vector_space_descriptor<Is>(t) == dim_bestI)))
        return std::optional {dim_bestI};
      else
        return std::optional<decltype(dim_bestI)> {};
    }
  }


  /**
   * \brief Return true if T is a \ref square_matrix at runtime.
   * \tparam T A tensor or matrix
   * \return a \ref std::optional which includes the \ref vector_space_descriptor object if T is square.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr auto get_is_square(const T& t)
  {
    if constexpr (square_matrix<T>)
      return std::optional {get_vector_space_descriptor<detail::get_best_square_index<T>()>(t)};
    else if constexpr (not square_matrix<T, Likelihood::maybe>)
      return std::optional<std::size_t> {};
    else if constexpr (index_count_v<T> == 1 and dimension_size_of_index_is<T, 0, 1, Likelihood::maybe>)
    {
      auto d = get_vector_space_descriptor<0>(t);
      if (get_dimension_size_of(d) == 1) return std::optional {d};
      else return std::optional<decltype(d)> {};
    }
    else return detail::get_is_square_impl(std::make_index_sequence<index_count_v<T>>{}, t);
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
    else return detail::get_is_one_by_one_impl(std::make_index_sequence<index_count_v<T>>{}, t);
  }


  // --------------- //
  //  get_is_vector  //
  // --------------- //

  namespace detail
  {
    template<std::size_t N, std::size_t...Is, typename T>
    constexpr bool get_is_vector_impl(std::index_sequence<Is...>, const T& t)
    {
      return (... and (N == Is or get_index_dimension_of<Is>(t) == 1));
    }
  }


  /**
   * \brief Return true if T is a \ref square_matrix at runtime.
   * \tparam N An index designating the "large" index (0 for a column vector, 1 for a row vector)
   * \tparam T A tensor or matrix
   * \sa vector
   */
#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible T>
#else
  template<std::size_t N = 0, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
  constexpr bool get_is_vector(const T& t)
  {
    if constexpr (vector<T, N>) return true;
    else if constexpr (not vector<T, N, Likelihood::maybe>) return false;
    else return detail::get_is_vector_impl<N>(std::make_index_sequence<index_count_v<T>>{}, t);
  }


  // --------------- //
  //  nested_matrix  //
  // --------------- //

  /**
   * \brief Retrieve a nested matrix of Arg, if it exists.
   * \tparam i Index of the nested matrix (0 for the 1st, 1 for the 2nd, etc.).
   * \tparam Arg A wrapper that has at least one nested matrix.
   * \internal \sa interface::indexible_object_traits::get_nested_matrix
   */
#ifdef __cpp_concepts
  template<std::size_t i = 0, typename Arg> requires
    (i < std::tuple_size_v<typename interface::indexible_object_traits<std::decay_t<Arg>>::type>) and
    requires(Arg&& arg) { interface::indexible_object_traits<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg)); }
#else
  template<std::size_t i = 0, typename Arg,
    std::enable_if_t<(i < std::tuple_size<typename interface::indexible_object_traits<std::decay_t<Arg>>::type>::value), int> = 0,
    typename = std::void_t<decltype(interface::indexible_object_traits<std::decay_t<Arg>>::template get_nested_matrix<i>(std::declval<Arg&&>()))>>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg)
  {
    return interface::indexible_object_traits<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg));
  }


  // --------------- //
  //  get_wrappable  //
  // --------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool get_wrappable_impl(const T& t, std::index_sequence<I...>)
    {
      return (get_vector_space_descriptor_is_euclidean(get_vector_space_descriptor<I + 1>(t)) and ...);
    }
  }


  /**
   * \brief Determine whether T is wrappable (i.e., all its dimensions other than potentially 0 are euclidean).
   * \tparam T A matrix or array
   * \todo Is this necessary?
   * \sa wrappable
   */
#ifdef __cpp_concepts
  template<indexible T> requires (index_count_v<T> >= 1)
#else
  template<typename T, std::enable_if_t<indexible<T> and (index_count_v<T> >= 1), int> = 0>
#endif
  constexpr bool get_wrappable(const T& t)
  {
    return detail::get_wrappable_impl(t, std::make_index_sequence<index_count_v<T> - 1> {});
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
     * \internal \sa interface::indexible_object_traits
     */
#ifdef __cpp_concepts
    template<std::size_t N, indexible T>
#else
    template<std::size_t N, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
    constexpr auto index_dimension_scalar_constant_of(const T& t)
    {
      using Scalar = scalar_type_of_t<T>;
      if constexpr (dynamic_dimension<T, N>) return static_cast<Scalar>(get_index_dimension_of<N>(t));
      else return ScalarConstant<Likelihood::definitely, Scalar, index_dimension_of_v<T, N>>{};
    }


    // -------------------------- //
    //  index_dimension_value_of  //
    // -------------------------- //

    /**
     * \internal
     * \brief Returns an \ref index_value reflecting the size of an index for a tensor or matrix.
     * \details The return value is a fixed or dynamic \ref index_value.
     * \tparam N The index
     * \tparam T The matrix, expression, or array
     */
#ifdef __cpp_concepts
    template<std::size_t N, indexible T>
#else
    template<std::size_t N, typename T, std::enable_if_t<indexible<T>, int> = 0>
#endif
    constexpr auto index_dimension_value_of(const T& t)
    {
      if constexpr (dynamic_dimension<T, N>) return get_index_dimension_of<N>(t);
      else return std::integral_constant<std::size_t, index_dimension_of_v<T, N>>{};
    }


    // ---------- //
    //  raw_data  //
    // ---------- //

    /**
     * \internal
     * \brief Returns a pointer to the raw data of a directly accessible tensor or matrix.
     */
#ifdef __cpp_concepts
    template<directly_accessible T>
#else
    template<typename T, std::enable_if_t<directly_accessible<T>, int> = 0>
#endif
    constexpr auto* raw_data(T& t)
    {
      return interface::indexible_object_traits<std::decay_t<T>>::data(t);
    }


    // --------- //
    //  strides  //
    // --------- //

    namespace detail
    {
      template<Layout l, typename T, typename CurrStride, std::size_t I, std::size_t...Is, typename...Strides>
      constexpr auto strides_impl(const T& t, CurrStride curr_stride, std::index_sequence<I, Is...>, Strides...strides)
      {
        if constexpr (sizeof...(Is) == 0)
        {
          if constexpr (l == Layout::right)
            return std::tuple {curr_stride, strides...};
          else
            return std::tuple {strides..., curr_stride};
        }
        else
        {
          auto curr_dim = static_cast<std::ptrdiff_t>(get_index_dimension_of<l == Layout::right ? index_count_v<T> - 1 - I : I>(t));
          auto next_stride = [](CurrStride curr_stride, auto curr_dim)
          {
            if constexpr (static_index_value<CurrStride> and static_index_value<decltype(curr_dim)>)
              return std::integral_constant<std::size_t, static_index_value_of_v<CurrStride> * static_index_value_of_v<decltype(curr_dim)>>{};
            else
              return curr_stride * curr_dim;
          }(curr_stride, curr_dim);

          if constexpr (l == Layout::right)
            return strides_impl<l>(t, next_stride, std::index_sequence<Is...>{}, curr_stride, strides...);
          else
            return strides_impl<l>(t, next_stride, std::index_sequence<Is...>{}, strides..., curr_stride);
        }
      }


#if __cpp_generic_lambdas >= 201707L
      template<typename T>
      concept valid_stride_tuple = []<std::size_t...ix>(std::index_sequence<ix...>){
        return ((std::is_same_v<std::decay_t<std::tuple_element_t<ix, T>>, std::ptrdiff_t> or
          std::is_same_v<typename static_index_value_of<typename std::tuple_element_t<ix, T>>::value_type, std::ptrdiff_t>) and ...);
        }(std::make_index_sequence<index_count_v<T>>{});
#endif

    } // namespace detail


    /**
     * \internal
     * \brief Returns a tuple <code>std::tuple&lt;S...&gt;</code> comprising the strides of a strided tensor or matrix.
     * \details Each of the strides <code>S</code> satisfies one of the two concepts:
     * - <code>std::same_as&lt;std::decay_t<S>, std::ptrdiff_t*gt;</code>; or
     * - <code>std::same_as&lt;typename static_index_value_of&lt;S&gt;>::value_type, std::ptrdiff_t*gt;</code>; or
     */
#if __cpp_generic_lambdas >= 201707L
    template<indexible T> requires (layout_of_v<T> != Layout::none)
    constexpr detail::valid_stride_tuple auto
#else
    template<typename T, std::enable_if_t<indexible<T> and layout_of<T>::value != Layout::none, int> = 0>
    constexpr auto
#endif
    strides(const T& t)
    {
      constexpr auto l = layout_of_v<T>;

      if constexpr (l == Layout::stride)
      {
        return interface::indexible_object_traits<std::decay_t<T>>::strides(t);
      }
      else
      {
        constexpr std::integral_constant<std::ptrdiff_t, 1> N1;
        return detail::strides_impl<l>(t, N1, std::make_index_sequence<index_count_v<T>>{});
      }
    }

    // -------------------- //
    //  has_static_strides  //
    // -------------------- //

    namespace detail
    {
      template<typename Strides, std::size_t...ix>
      constexpr bool has_static_strides_i(std::index_sequence<ix...>)
      {
        return (static_index_value<std::tuple_element_t<ix, Strides>> and ...);
      };

      template<typename Strides>
      constexpr bool has_static_strides_impl()
      {
        return has_static_strides_i<Strides>(std::make_index_sequence<std::tuple_size_v<Strides>>{});
      };
    }


    /**
     * \brief Specifies that T has strides that are known at compile time.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept has_static_strides =
#else
    constexpr bool has_static_strides =
#endif
      detail::has_static_strides_impl<decltype(internal::strides(std::declval<T>()))>();

  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_INDEXIBLE_PROPERTY_FUNCTIONS_HPP
