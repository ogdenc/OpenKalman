/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Functions for \ref vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_FUNCTIONS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_FUNCTIONS_HPP

#include <type_traits>

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif

namespace OpenKalman
{
  // -------------- //
  //   Comparison   //
  // -------------- //

  namespace internal
  {
    template<typename T>
    struct is_DynamicTypedIndex : std::false_type {};

    template<typename...AllowableScalarTypes>
    struct is_DynamicTypedIndex<DynamicTypedIndex<AllowableScalarTypes...>> : std::true_type {};
  } // namespace detail


#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  /**
   * \brief Three-way comparison for a non-built-in \ref vector_space_descriptor.
   * \details A comparison of A and B is a partial ordering based on whether or not A is a prefix of B.
   */
  template<vector_space_descriptor A, vector_space_descriptor B>
  constexpr auto operator<=>(const A& a, const B& b) requires (not std::integral<A>) or (not std::integral<B>)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
    {
      if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
        return dimension_size_of_v<A> <=> dimension_size_of_v<B>;
      else if constexpr (equivalent_to<A, B>)
        return std::partial_ordering::equivalent;
      else if constexpr (prefix_of<A, B>)
        return std::partial_ordering::less;
      else if constexpr (prefix_of<B, A>)
        return std::partial_ordering::greater;
      else
        return std::partial_ordering::unordered;
    }
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
    {
      return get_dimension_size_of(a) <=> get_dimension_size_of(b);
    }
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
    {
      if (a.partially_matches(b)) return std::partial_ordering {get_dimension_size_of(a) <=> get_dimension_size_of(b)};
      else return std::partial_ordering::unordered;
    }
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
    {
      if (b.partially_matches(a)) return std::partial_ordering {get_dimension_size_of(a) <=> get_dimension_size_of(b)};
      else return std::partial_ordering::unordered;
    }
    else
    {
      return std::partial_ordering::unordered;
    }
  }


  /**
   * \brief Equality comparison for non-built-in \ref vector_space_descriptor.
   */
  constexpr bool operator==(const vector_space_descriptor auto& a, const vector_space_descriptor auto& b)
    requires (not std::integral<decltype(a)>) or (not std::integral<decltype(b)>)
  {
    return std::is_eq(a <=> b);
  }
#else
  /**
   * \brief Equivalence comparison for a non-built-in \ref vector_space_descriptor.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
      (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return equivalent_to<A, B>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return get_dimension_size_of(a) == get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partially_matches(b) and get_dimension_size_of(a) == get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partially_matches(a) and get_dimension_size_of(a) == get_dimension_size_of(b);
    else
      return false;
  }


  /**
   * \brief Compares \ref vector_space_descriptor objects for non-equivalence.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is less than another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return prefix_of<A, B>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return get_dimension_size_of(a) < get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partially_matches(b) and get_dimension_size_of(a) < get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partially_matches(a) and get_dimension_size_of(a) < get_dimension_size_of(b);
    else
      return false;
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is greater than another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<B>)
      return prefix_of<B, A>;
    else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<B>)
      return get_dimension_size_of(a) > get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<A>::value)
      return a.partially_matches(b) and get_dimension_size_of(a) > get_dimension_size_of(b);
    else if constexpr (internal::is_DynamicTypedIndex<B>::value)
      return b.partially_matches(a) and get_dimension_size_of(a) > get_dimension_size_of(b);
    else
      return false;
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is less than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    return operator<(a, b) or operator==(a, b);
  }


  /**
   * \brief Determine whether one \ref vector_space_descriptor object is greater than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<B> and
    (not std::is_integral_v<A> or not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    return operator>(a, b) or operator==(a, b);
  }
#endif


  // -------------- //
  //   Arithmetic   //
  // -------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
    concept vector_space_descriptor_arithmetic_defined =
      interface::FixedVectorSpaceDescriptorTraits<T>::operations_defined or interface::DynamicVectorSpaceDescriptorTraits<T>::operations_defined;
#else
    template<typename T, typename = void>
    struct fixed_vector_space_descriptor_arithmetic_defined : std::false_type {};

    template<typename T>
    struct fixed_vector_space_descriptor_arithmetic_defined<T, std::enable_if_t<
      interface::FixedVectorSpaceDescriptorTraits<T>::operations_defined>> : std::true_type {};

    template<typename T, typename = void>
    struct dynamic_vector_space_descriptor_arithmetic_defined : std::false_type {};

    template<typename T>
    struct dynamic_vector_space_descriptor_arithmetic_defined<T, std::enable_if_t<
      interface::DynamicVectorSpaceDescriptorTraits<T>::operations_defined>> : std::true_type {};

    template<typename T>
    constexpr bool vector_space_descriptor_arithmetic_defined =
      fixed_vector_space_descriptor_arithmetic_defined<T>::value or dynamic_vector_space_descriptor_arithmetic_defined<T>::value;
#endif
  } // namespace detail


  /**
   * \brief Add two sets of \ref vector_space_descriptor, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, vector_space_descriptor U> requires
    detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>
#else
  template<typename T, typename U, std::enable_if_t<vector_space_descriptor<T> and vector_space_descriptor<U> and
    (detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>), int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (fixed_vector_space_descriptor<T> and fixed_vector_space_descriptor<U>)
    {
      return concatenate_fixed_vector_space_descriptor_t<T, U> {};
    }
    else if constexpr (euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U>)
    {
      if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
        return Dimensions{get_dimension_size_of(t) + get_dimension_size_of(u)};
      else
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
    }
    else
    {
      return DynamicTypedIndex {std::forward<T>(t), std::forward<U>(u)};
    }
  }


  /**
   * \brief Subtract two \ref euclidean_vector_space_descriptor values, whether fixed or dynamic.
   * \warning This does not perform any runtime checks to ensure that the result is non-negative.
   */
#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T, euclidean_vector_space_descriptor U> requires (dimension_size_of_v<T> == dynamic_size or
      dimension_size_of_v<U> == dynamic_size or dimension_size_of_v<T> > dimension_size_of_v<U>) and
    (detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>)
#else
  template<typename T, typename U, std::enable_if_t<euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U> and
    ((dimension_size_of<T>::value == dynamic_size) or (dimension_size_of<U>::value == dynamic_size) or
      (dimension_size_of<T>::value > dimension_size_of<U>::value)) and
      (detail::vector_space_descriptor_arithmetic_defined<T> or detail::vector_space_descriptor_arithmetic_defined<U>), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u) noexcept
  {
    if constexpr (dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U>)
      return Dimensions{get_dimension_size_of(t) - get_dimension_size_of(u)};
    else
      return Dimensions<dimension_size_of_v<T> - dimension_size_of_v<U>>{};
  }


  namespace internal
  {

    // ------------------------------------- //
    //   replicate_vector_space_descriptor   //
    // ------------------------------------- //

    /**
     * \brief Replicate \ref fixed_vector_space_descriptor T some number of times.
     */
#ifdef __cpp_concepts
    template<scalar_type..., fixed_vector_space_descriptor T, static_index_value N>
#else
    template<typename...S, typename T, typename N, std::enable_if_t<(scalar_type<S> and ...) and
      fixed_vector_space_descriptor<T> and static_index_value<N>, int> = 0>
#endif
    auto replicate_vector_space_descriptor(const T& t, N n)
    {
      return replicate_fixed_vector_space_descriptor_t<T, n> {};
    }


    /**
     * \overload
     * \brief Replicate a (potentially) dynamic \ref vector_space_descriptor T some (potentially) dynamic number of times.
     */
#ifdef __cpp_concepts
    template<scalar_type...AllowableScalarTypes, vector_space_descriptor T, index_value N>
    requires (not fixed_vector_space_descriptor<T>) or (not static_index_value<N>)
#else
    template<typename...AllowableScalarTypes, typename T, typename N, std::enable_if_t<
      (scalar_type<AllowableScalarTypes> and ...) and vector_space_descriptor<T> and index_value<N> and
        (not fixed_vector_space_descriptor<T> or not static_index_value<N>), int> = 0>
#endif
    auto replicate_vector_space_descriptor(const T& t, N n)
    {
      auto ret = [](const T& t){
        if constexpr (sizeof...(AllowableScalarTypes) > 0) return DynamicTypedIndex<AllowableScalarTypes...> {t};
        else return DynamicTypedIndex {t};
      }(t);
      for (std::size_t i = 1; i < static_cast<std::size_t>(n); ++i) ret.extend(t);
      return ret;
    }


    // --------------------------- //
    //   is_uniform_component_of   //
    // --------------------------- //

    /**
     * \internal
     * \brief Whether <code>a</code> is a 1D \ref vector_space_descriptor object that, when replicated some number of times, becomes <code>c</code>.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor A, vector_space_descriptor C>
#else
    template<typename A, typename C, std::enable_if_t<vector_space_descriptor<A> and vector_space_descriptor<C>, int> = 0>
#endif
    constexpr bool is_uniform_component_of(const A& a, const C& c)
    {
      if constexpr (fixed_vector_space_descriptor<A> and fixed_vector_space_descriptor<C>)
        return equivalent_to_uniform_dimension_type_of<A, C>;
      else if constexpr (euclidean_vector_space_descriptor<A> and euclidean_vector_space_descriptor<C>)
        return get_dimension_size_of(a) == 1;
      else
        return false;
    }


    /**
     * \internal
     * \overload
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor A, typename...S>
#else
    template<typename A, typename...S, std::enable_if_t<vector_space_descriptor<A>, int> = 0>
#endif
    constexpr bool is_uniform_component_of(const A& a, const DynamicTypedIndex<S...>& c)
    {
      if (get_dimension_size_of(a) != 1) return false;
      else if (get_vector_space_descriptor_is_euclidean(a) and get_vector_space_descriptor_is_euclidean(c)) return true;
      else return replicate_vector_space_descriptor<S...>(a, get_dimension_size_of(c)) == c;
    }


    /**
     * \internal
     * \overload
     */
#ifdef __cpp_concepts
    template<typename...T, vector_space_descriptor C>
#else
    template<typename...T, typename C, std::enable_if_t<vector_space_descriptor<C>, int> = 0>
#endif
    constexpr bool is_uniform_component_of(const DynamicTypedIndex<T...>& a, const C& c)
    {
      if (get_dimension_size_of(a) != 1) return false;
      else if (get_vector_space_descriptor_is_euclidean(a) and get_vector_space_descriptor_is_euclidean(c)) return true;
      else return replicate_vector_space_descriptor(a, get_dimension_size_of(c)) == c;
    }


    /**
     * \internal
     * \overload
     */
    template<typename...T, typename...S>
    constexpr bool is_uniform_component_of(const DynamicTypedIndex<T...>& a, const DynamicTypedIndex<S...>& c)
    {
      if constexpr (((not std::is_same_v<T, S>) or ...)) return false;
      else if (get_dimension_size_of(a) != 1) return false;
      else if (get_vector_space_descriptor_is_euclidean(a) and get_vector_space_descriptor_is_euclidean(c)) return true;
      else return replicate_vector_space_descriptor(a, get_dimension_size_of(c)) == c;
    }


    // ---------------------------------- //
    //   remove_trailing_1D_descriptors   //
    // ---------------------------------- //

    /**
     * \brief Remove any trailing, one-dimensional \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    template<tuple_like DTup>
#else
    template<typename DTup, std::enable_if_t<tuple_like<DTup>, int> = 0>
#endif
    constexpr auto remove_trailing_1D_descriptors(DTup&& d_tup)
    {
      constexpr auto N = std::tuple_size_v<DTup>;
      if constexpr (N == 0)
        return std::forward<DTup>(d_tup);
      else if constexpr (equivalent_to<std::tuple_element_t<N - 1, std::decay_t<DTup>>, Dimensions<1>>)
        return remove_trailing_1D_descriptors(tuple_slice<0, N - 1>(std::forward<DTup>(d_tup)));
      else
        return std::forward<DTup>(d_tup);
    }

  } // namespace internal


} // namespace OpenKalman


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_FUNCTIONS_HPP
