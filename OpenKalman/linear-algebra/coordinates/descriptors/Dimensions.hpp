/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
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
#include <type_traits>
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"

namespace OpenKalman::coordinates
{
  // ------------ //
  //  fixed case  //
  // ------------ //

  /**
   * \brief A structure representing the dimensions associated with of a particular index.
   * \details The dimension may or may not be known at compile time. If unknown at compile time, the size is set
   * at the time of construction and cannot be modified thereafter.
   * \tparam N The dimension (or <code>dynamic_size</code>, if not known at compile time)
   */
  template<std::size_t N = dynamic_size>
  struct Dimensions
  {
    /// Default constructor
    constexpr Dimensions() = default;


    /// Constructor, taking a static \ref coordinates::euclidean_pattern.
#ifdef __cpp_concepts
    template<typename D> requires (not std::same_as<std::decay_t<D>, Dimensions>) and
      ((euclidean_pattern<D> and fixed_pattern<D> and dimension_of_v<D> == N) or
      dynamic_pattern<D>)
#else
    template<typename D, std::enable_if_t<
      (not std::is_same_v<std::decay_t<D>, Dimensions>) and
      ((euclidean_pattern<D> and fixed_pattern<D> and dimension_of<D>::value == N) or
      dynamic_pattern<D>), int> = 0>
#endif
    explicit constexpr Dimensions(D&& d)
    {
      if constexpr (dynamic_pattern<D>)
      {
        if constexpr (not euclidean_pattern<D>)
        {
          if (not get_is_euclidean(d))
            throw std::invalid_argument{"Argument of dynamic 'Dimensions' constructor must be a euclidean_coordinate_list."};
        }
        if (get_dimension(d) != N)
          throw std::invalid_argument{"Dynamic argument to static 'Dimensions' constructor has the wrong size."};
      }
    }


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
  struct Dimensions<dynamic_size>
  {
    /// Construct from a \ref coordinates::euclidean_pattern or \ref dynamic_pattern.
#ifdef __cpp_concepts
    template<typename D> requires (euclidean_pattern<D> or dynamic_pattern<D>) and
      (not std::is_base_of_v<Dimensions, D>)
#else
    template<typename D, std::enable_if_t<(euclidean_pattern<D> or dynamic_pattern<D>) and
      (not std::is_base_of_v<Dimensions, D>), int> = 0>
#endif
    explicit constexpr Dimensions(const D& d) : runtime_size {get_dimension(d)}
    {
      if constexpr (not euclidean_pattern<D>)
        if (not get_is_euclidean(d))
          throw std::invalid_argument{"Argument of dynamic 'Dimensions' constructor must be a euclidean_coordinate_list."};
    }


    /// Construct from an integral value.
    explicit constexpr Dimensions(const std::size_t& d = 0) : runtime_size {static_cast<std::size_t>(d)}
    {}


    /**
     * \brief Assign from another \ref coordinates::euclidean_pattern or \ref dynamic_pattern.
     */
#ifdef __cpp_concepts
    template<typename D> requires (euclidean_pattern<D> or dynamic_pattern<D>) and
      (not std::is_base_of_v<Dimensions, D>)
#else
    template<typename D, std::enable_if_t<(euclidean_pattern<D> or dynamic_pattern<D>) and
      (not std::is_base_of_v<Dimensions, D>), int> = 0>
#endif
    constexpr Dimensions& operator=(const D& d)
    {
      if constexpr (not euclidean_pattern<D>)
        if (not get_is_euclidean(d))
          throw std::invalid_argument{"Argument of dynamic 'Dimensions' assignment operator must be a euclidean_coordinate_list."};
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

    friend struct interface::coordinate_descriptor_traits<Dimensions>;
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
  explicit Dimensions(D&&) -> Dimensions<dynamic_size>;


  explicit Dimensions(const std::size_t&) -> Dimensions<dynamic_size>;


  // ------ //
  //  Axis  //
  // ------ //

  /**
   * \brief Alias for a 1D euclidean \ref coordinates::pattern object.
   */
  using Axis = Dimensions<1>;


} // OpenKalman::coordinates


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Dimensions.
   */
  template<std::size_t N>
  struct coordinate_descriptor_traits<coordinates::Dimensions<N>>
  {
  private:

    using T = coordinates::Dimensions<N>;

  public:

    static constexpr bool is_specialized = true;


    static constexpr auto
    dimension(const T& t)
    {
      if constexpr (N == dynamic_size) return t.runtime_size;
      else return std::integral_constant<std::size_t, N>{};
    };


    static constexpr auto
    stat_dimension(const T& t) { return size(t); };


    static constexpr auto
    is_euclidean(const T&) { return std::true_type{}; }


    static constexpr std::size_t
    hash_code(const T& t)
    {
      if constexpr (N == dynamic_size)
      {
        return static_cast<std::size_t>(t.runtime_size);
      }
      else
      {
        return N;
      }
    }

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_DIMENSIONS_HPP
