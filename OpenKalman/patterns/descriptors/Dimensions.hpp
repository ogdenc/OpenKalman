/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Dimensions class.
 */

#ifndef OPENKALMAN_DIMENSIONS_HPP
#define OPENKALMAN_DIMENSIONS_HPP

#include <cstddef>
#include "collections/collections.hpp"
#include "patterns/interfaces/pattern_descriptor_traits.hpp"
#include "patterns/concepts/fixed_pattern.hpp"
#include "patterns/concepts/dynamic_pattern.hpp"
#include "patterns/concepts/euclidean_pattern.hpp"
#include "patterns/traits/dimension_of.hpp"
#include "patterns/functions/get_dimension.hpp"
#include "Any.hpp"

namespace OpenKalman::patterns
{
  // ------------ //
  //  fixed case  //
  // ------------ //

  /**
   * \brief A structure representing the dimensions associated with of a particular index.
   * \details The dimension may or may not be known at compile time. If unknown at compile time, the size is set
   * at the time of construction and cannot be modified thereafter.
   * \tparam N The dimension (or <code>stdex::dynamic_extent</code>, if not known at compile time)
   */
  template<std::size_t N = stdex::dynamic_extent>
  struct Dimensions
  {
    /// Default constructor
    constexpr Dimensions() = default;


    /// Constructor, taking a fixed-dimension \ref euclidean_pattern.
#ifdef __cpp_concepts
    template<euclidean_pattern D> requires
      (dimension_of<D>::value == N) and
      (not std::same_as<D, Dimensions>)
#else
    template<typename D, std::enable_if_t<
      euclidean_pattern<D> and
      values::fixed_value_compares_with<dimension_of<D>, N> and
      not std::is_same_v<D, Dimensions>, int> = 0>
#endif
    constexpr Dimensions(const D&) {}


    template<typename Int>
    explicit constexpr operator std::integral_constant<Int, N>()
    {
      return std::integral_constant<Int, N>{};
    }


#ifdef __cpp_concepts
    template<std::integral Int>
#else
    template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
#endif
    explicit constexpr operator Int()
    {
      return N;
    }

  }; // struct Dimensions, static case


  // -------------- //
  //  dynamic case  //
  // -------------- //

  /**
   * \overload
   * \brief Case where the dimension or size associated with a given dynamic index is known only at runtime.
   */
  template<>
  struct Dimensions<stdex::dynamic_extent>
  {
    /// Construct from a \ref patterns::euclidean_pattern or \ref dynamic_pattern.
#ifdef __cpp_concepts
    template<euclidean_pattern D> requires (not std::same_as<D, Dimensions>)
#else
    template<typename D, std::enable_if_t<euclidean_pattern<D> and (not std::is_same_v<Dimensions, D>), int> = 0>
#endif
    constexpr Dimensions(const D& d) : runtime_size {get_dimension(d)}
    {}


    /// Construct from an integral value.
    constexpr Dimensions(const std::size_t& d = 0) : runtime_size {static_cast<std::size_t>(d)}
    {}


    /**
     * \brief Assign from another \ref patterns::euclidean_pattern or \ref dynamic_pattern.
     */
#ifdef __cpp_concepts
    template<euclidean_pattern D> requires (not std::same_as<D, Dimensions>)
#else
    template<typename D, std::enable_if_t<euclidean_pattern<D> and (not std::is_same_v<D, Dimensions>), int> = 0>
#endif
    constexpr Dimensions& operator=(const D& d)
    {
      runtime_size = get_dimension(d);
      return *this;
    }


#ifdef __cpp_concepts
    template<std::integral Int>
#else
    template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
#endif
    explicit constexpr operator Int()
    {
      return runtime_size;
    }

  protected:

    std::size_t runtime_size;

    friend struct interface::pattern_descriptor_traits<Dimensions>;
  };


  // ------------------ //
  //  deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<fixed_pattern D> requires euclidean_pattern<D>
#else
  template<typename D, std::enable_if_t<fixed_pattern<D> and euclidean_pattern<D>, int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<dimension_of<D>::value>;


#ifdef __cpp_concepts
  template<dynamic_pattern D> requires euclidean_pattern<D>
#else
  template<typename D, std::enable_if_t<dynamic_pattern<D> and euclidean_pattern<D>, int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<stdex::dynamic_extent>;


  explicit Dimensions(const std::size_t&) -> Dimensions<stdex::dynamic_extent>;


  // ------ //
  //  Axis  //
  // ------ //

  /**
   * \brief Alias for a 1D Euclidean \ref patterns::pattern object.
   */
  using Axis = Dimensions<1>;

}


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Dimensions.
   */
  template<std::size_t N>
  struct pattern_descriptor_traits<patterns::Dimensions<N>>
  {
  private:

    using T = patterns::Dimensions<N>;

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto
    dimension = [](const T& t)
    {
      if constexpr (N == stdex::dynamic_extent) return t.runtime_size;
      else return std::integral_constant<std::size_t, N>{};
    };


    static constexpr auto
    stat_dimension = [](const T& t) { return dimension(t); };


    static constexpr auto
    is_euclidean = [](const T&) { return std::true_type{}; };


    static constexpr auto
    hash_code = [](const T& t)
    {
      if constexpr (N == stdex::dynamic_extent)
        return static_cast<std::size_t>(t.runtime_size);
      else
        return std::integral_constant<std::size_t, N>{};
    };

  };

}


namespace std
{
  template<std::size_t M, std::size_t N>
  struct common_type<OpenKalman::patterns::Dimensions<M>, OpenKalman::patterns::Dimensions<N>>
  {
    using type = OpenKalman::patterns::Dimensions<M == N ? N : OpenKalman::stdex::dynamic_extent>;
  };


  template<std::size_t N, typename T>
  struct common_type<OpenKalman::patterns::Dimensions<N>, T>
    : std::conditional_t<
      OpenKalman::patterns::descriptor<T>,
      std::conditional<OpenKalman::patterns::euclidean_pattern<T>,
        OpenKalman::patterns::Dimensions<N == OpenKalman::patterns::dimension_of_v<T> ? N : OpenKalman::stdex::dynamic_extent>,
        OpenKalman::patterns::Any<>>,
      std::monostate> {};


  template<std::size_t N, typename Scalar>
  struct common_type<OpenKalman::patterns::Dimensions<N>, OpenKalman::patterns::Any<Scalar>>
  {
    using type = OpenKalman::patterns::Any<Scalar>;
  };


  template<std::size_t N, typename Scalar>
  struct common_type<OpenKalman::patterns::Any<Scalar>, OpenKalman::patterns::Dimensions<N>>
  {
    using type = OpenKalman::patterns::Any<Scalar>;
  };


#ifdef __cpp_concepts
  template<OpenKalman::patterns::euclidean_pattern T, std::size_t N>
  struct common_type<T, OpenKalman::patterns::Dimensions<N>> : common_type<OpenKalman::patterns::Dimensions<N>, T> {};
#endif
}

#ifndef __cpp_concepts
#define OPENKALMAN_INDEX_STD_COMMON_TYPE_SPECIALIZATION(I)          \
namespace std                                                       \
{                                                                   \
  template<std::size_t N>                                           \
  struct common_type<I, OpenKalman::patterns::Dimensions<N>>     \
    : common_type<OpenKalman::patterns::Dimensions<N>, I> {};    \
}

OPENKALMAN_INDEX_STD_COMMON_TYPE_SPECIALIZATION(unsigned char)
OPENKALMAN_INDEX_STD_COMMON_TYPE_SPECIALIZATION(unsigned short)
OPENKALMAN_INDEX_STD_COMMON_TYPE_SPECIALIZATION(unsigned int)
OPENKALMAN_INDEX_STD_COMMON_TYPE_SPECIALIZATION(unsigned long)
OPENKALMAN_INDEX_STD_COMMON_TYPE_SPECIALIZATION(unsigned long long)

namespace std
{
  template<typename T, auto M, std::size_t N>
  struct common_type<std::integral_constant<T, M>, OpenKalman::patterns::Dimensions<N>>
    : common_type<OpenKalman::patterns::Dimensions<N>, std::integral_constant<T, M>> {};


  template<typename T, auto...M, std::size_t N>
  struct common_type<OpenKalman::values::fixed_value<T, M...>, OpenKalman::patterns::Dimensions<N>>
    : common_type<OpenKalman::patterns::Dimensions<N>, OpenKalman::values::fixed_value<T, M...>> {};


  template<typename...Args, std::size_t N>
  struct common_type<OpenKalman::values::consteval_operation<Args...>, OpenKalman::patterns::Dimensions<N>>
    : common_type<OpenKalman::patterns::Dimensions<N>, OpenKalman::values::consteval_operation<Args...>> {};
}
#endif


#endif
