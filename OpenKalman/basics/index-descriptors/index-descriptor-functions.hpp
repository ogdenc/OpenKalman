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
 * \brief Functions for accessing elements of typed arrays, based on typed coefficients.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP

#include <type_traits>
#include <functional>

#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif

namespace OpenKalman
{
  // ------------------------- //
  //   get_dimension_size_of   //
  // ------------------------- //

  /**
   * \brief Get the dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_dimension_size_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return dimension_size_of_v<T>;
    else return interface::IndexDescriptorSize<T>::get(t);
  }


  // ----------------------------------- //
  //   get_euclidean_dimension_size_of   //
  // ----------------------------------- //

  /**
   * \brief Get the Euclidean dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_euclidean_dimension_size_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return euclidean_dimension_size_of_v<T>;
    else return interface::EuclideanIndexDescriptorSize<T>::get(t);
  }


  // ------------------------------------------- //
  //   get_index_descriptor_component_count_of   //
  // ------------------------------------------- //

  /**
   * \brief Get the dimensions of \ref index_descriptor T
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr std::size_t
  get_index_descriptor_component_count_of(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return index_descriptor_components_of_v<T>;
    else return interface::IndexDescriptorComponentCount<T>::get(t);
  }


  // ----------------------------------- //
  //   get_index_descriptor_is_untyped   //
  // ----------------------------------- //

  /**
   * \brief Determine, at runtime, whether \ref index_descriptor T is untyped.
   */
#ifdef __cpp_concepts
  template<index_descriptor T>
#else
  template<typename T, std::enable_if_t<index_descriptor<T>, int> = 0>
#endif
  constexpr bool
  get_index_descriptor_is_untyped(const T& t)
  {
    if constexpr (fixed_index_descriptor<T>) return euclidean_index_descriptor<T>;
    else return interface::IndexDescriptorIsUntyped<T>::get(t);
  }


  // -------------- //
  //   Comparison   //
  // -------------- //

#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  /**
   * \brief Three-way comparison for a non-built-in \ref index_descriptor.
   */
  template<index_descriptor A, index_descriptor B> requires (not std::integral<A>) and (not std::integral<B>)
  constexpr auto operator<=>(const A& a, const B& b)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
    {
      if constexpr (dimension_size_of_v<A> == dimension_size_of_v<B>)
      {
        if constexpr (equivalent_to<A, B>) return std::partial_ordering::equivalent;
        else return std::partial_ordering::unordered;
      }
      else return std::partial_ordering {dimension_size_of_v<A> <=> dimension_size_of_v<B>};
    }
    else if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<B>)
    {
      return get_dimension_size_of(a) <=> get_dimension_size_of(b);
    }
    else // At least one of A or B is dynamic (DynamicTypedIndex or Dimensions).
    {
      auto size_a = get_dimension_size_of(a);
      auto size_b = get_dimension_size_of(b);

      if (size_a == size_b)
      {
        if constexpr (dynamic_index_descriptor<A> and not euclidean_index_descriptor<A>)
        {
          if (a.is_equivalent(b)) return std::partial_ordering::equivalent;
          else return std::partial_ordering::unordered;
        }
        else if constexpr (dynamic_index_descriptor<B> and not euclidean_index_descriptor<B>)
        {
          if (b.is_equivalent(a)) return std::partial_ordering::equivalent;
          else return std::partial_ordering::unordered;
        }
        else
        {
          return std::partial_ordering::unordered;
        }
      }
      else
      {
        return std::partial_ordering {size_a <=> size_b};
      }
    }
  }


  /**
   * \brief Equality comparison for non-built-in \ref index_descriptors.
   */
  template<index_descriptor A, index_descriptor B> requires (not std::integral<A>) and (not std::integral<B>)
  constexpr bool operator==(const A& a, const B& b)
  {
    return std::is_eq(a <=> b);
  }
#else
  /**
   * \brief Equivalence comparison for a non-built-in \ref index_descriptor.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
      (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    if constexpr (fixed_index_descriptor<A> and fixed_index_descriptor<B>)
    {
      return equivalent_to<A, B>;
    }
    else if constexpr (euclidean_index_descriptor<A> and euclidean_index_descriptor<B>)
    {
      return get_dimension_size_of(a) == get_dimension_size_of(b);
    }
    else // At least one of A or B is dynamic (DynamicTypedIndex or Dimensions).
    {
      if (get_dimension_size_of(a) == get_dimension_size_of(b))
      {
        if constexpr (dynamic_index_descriptor<A> and not euclidean_index_descriptor<A>)
          return a.is_equivalent(b);
        else if constexpr (dynamic_index_descriptor<B> and not euclidean_index_descriptor<B>)
          return b.is_equivalent(a);
        else
          return false;
      }
      else
      {
        return false;
      }
    }
  }


  /**
   * \brief Compares index descriptors for non-equivalence.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return not operator==(a, b);
  }


  /**
   * \brief Determine whether one index descriptor is less than another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<(const A& a, const B& b)
  {
    return get_dimension_size_of(a) < get_dimension_size_of(b);
  }


  /**
   * \brief Determine whether one index descriptor is greater than another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>(const A& a, const B& b)
  {
    return get_dimension_size_of(a) > get_dimension_size_of(b);
  }


  /**
   * \brief Determine whether one index descriptor is less than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator<=(const A& a, const B& b)
  {
    return operator<(a, b) or operator==(a, b);
  }


  /**
   * \brief Determine whether one index descriptor is greater than or equal to another.
   */
  template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
    (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int> = 0>
  constexpr bool operator>=(const A& a, const B& b)
  {
    return operator>(a, b) or operator==(a, b);
  }
#endif


  // -------------- //
  //   Arithmetic   //
  // -------------- //

  /**
   * \brief Add two \ref index_descriptor values, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<index_descriptor T, index_descriptor U> requires
    (not (typed_index_descriptor<T> or typed_index_descriptor<U>) or (fixed_index_descriptor<T> and fixed_index_descriptor<U>))
#else
  template<typename T, typename U, std::enable_if_t<index_descriptor<T> and index_descriptor<U> and
    (not (typed_index_descriptor<T> or typed_index_descriptor<U>) or (fixed_index_descriptor<T> and fixed_index_descriptor<U>)), int> = 0>
#endif
  constexpr auto operator+(const T& t, const U& u) noexcept
  {
    if constexpr (typed_index_descriptor<T> or typed_index_descriptor<U>)
    {
      return concatenate_fixed_index_descriptor_t<T, U> {};
    }
    else
    {
      if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
        return Dimensions{get_dimension_size_of(t) + get_dimension_size_of(u)};
      else
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
    }
  }


  /**
   * \brief Subtract two \ref euclidean_index_descriptor values, whether fixed or dynamic.
   * \warning This does not perform any runtime checks to ensure that the result is non-negative.
   */
#ifdef __cpp_concepts
  template<euclidean_index_descriptor T, euclidean_index_descriptor U> requires (dimension_size_of_v<T> == dynamic_size) or
    (dimension_size_of_v<U> == dynamic_size) or (dimension_size_of_v<T> > dimension_size_of_v<U>)
#else
  template<typename T, typename U, std::enable_if_t<euclidean_index_descriptor<T> and euclidean_index_descriptor<U> and
    ((dimension_size_of<T>::value == dynamic_size) or (dimension_size_of<U>::value == dynamic_size) or
      (dimension_size_of<T>::value > dimension_size_of<U>::value)), int> = 0>
#endif
  constexpr auto operator-(const T& t, const U& u) noexcept
  {
    if constexpr (dimension_size_of_v<T> == dynamic_size or dimension_size_of_v<U> == dynamic_size)
      return Dimensions{get_dimension_size_of(t) - get_dimension_size_of(u)};
    else
      return Dimensions<dimension_size_of_v<T> - dimension_size_of_v<U>>{};
  }

} // namespace OpenKalman


#endif //OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
