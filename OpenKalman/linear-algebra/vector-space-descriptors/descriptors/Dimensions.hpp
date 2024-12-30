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
 * \brief Definition of the Dimensions class.
 */

#ifndef OPENKALMAN_DIMENSIONS_HPP
#define OPENKALMAN_DIMENSIONS_HPP

#include <cstddef>
#include <type_traits>
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp"
#include "StaticDescriptor.hpp"

namespace OpenKalman::descriptor
{
  // ------------ //
  //  fixed case  //
  // ------------ //

  /**
   * \brief Case in which the dimension or size associated with a given index is known at compile time.
   * \tparam N The dimension as known at compile time
   */
  template<std::size_t N>
  struct Dimensions
  {
    /// Default constructor
    constexpr Dimensions() = default;


    /// Constructor, taking a static \ref euclidean_vector_space_descriptor.
#ifdef __cpp_concepts
    template<typename D> requires (not std::same_as<std::decay_t<D>, Dimensions>) and
      ((euclidean_vector_space_descriptor<D> and static_vector_space_descriptor<D> and dimension_size_of_v<D> == N) or
      dynamic_vector_space_descriptor<D>)
#else
    template<typename D, std::enable_if_t<
      (not std::is_same_v<std::decay_t<D>, Dimensions>) and
      ((euclidean_vector_space_descriptor<D> and static_vector_space_descriptor<D> and dimension_size_of_v<D> == N) or
      dynamic_vector_space_descriptor<D>), int> = 0>
#endif
    explicit constexpr Dimensions(D&& d)
    {
      if constexpr (dynamic_vector_space_descriptor<D>)
      {
        if constexpr (not euclidean_vector_space_descriptor<D>)
        {
          if (not get_vector_space_descriptor_is_euclidean(d))
            throw std::invalid_argument{"Argument of dynamic 'Dimensions' constructor must be a euclidean vector space descriptor."};
        }
        if (get_dimension_size_of(d) != N)
          throw std::invalid_argument{"Dynamic argument to static 'Dimensions' constructor has the wrong size."};
      }
    }


    template<typename Int>
    constexpr operator std::integral_constant<Int, N>()
    {
      return std::integral_constant<Int, N>{};
    }


#ifdef __cpp_concepts
    template<std::integral Int>
#else
    template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
#endif
    constexpr operator Int()
    {
      return N;
    }

  }; // struct Dimensions, static case


  // -------------- //
  //  dynamic case  //
  // -------------- //

  /**
   * \brief Case where the dimension or size associated with a given dynamic index is known only at runtime.
   */
  template<>
  struct Dimensions<dynamic_size>
  {
    /// Construct from a \ref euclidean_vector_space_descriptor or \ref dynamic_vector_space_descriptor.
#ifdef __cpp_concepts
    template<typename D> requires (euclidean_vector_space_descriptor<D> or dynamic_vector_space_descriptor<D>) and
      (not std::is_base_of_v<Dimensions, D>)
#else
    template<typename D, std::enable_if_t<(euclidean_vector_space_descriptor<D> or dynamic_vector_space_descriptor<D>) and
      (not std::is_base_of_v<Dimensions, D>), int> = 0>
#endif
    explicit constexpr Dimensions(const D& d) : runtime_size {get_dimension_size_of(d)}
    {
      if constexpr (not euclidean_vector_space_descriptor<D>)
        if (not get_vector_space_descriptor_is_euclidean(d))
          throw std::invalid_argument{"Argument of dynamic 'Dimensions' constructor must be a euclidean vector space descriptor."};
    }


    /// Construct from an integral value.
    explicit constexpr Dimensions(const std::size_t& d = 0) : runtime_size {static_cast<std::size_t>(d)}
    {}


    /**
     * \brief Assign from another \ref euclidean_vector_space_descriptor or \ref dynamic_vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template<typename D> requires (euclidean_vector_space_descriptor<D> or dynamic_vector_space_descriptor<D>) and
      (not std::is_base_of_v<Dimensions, D>)
#else
    template<typename D, std::enable_if_t<(euclidean_vector_space_descriptor<D> or dynamic_vector_space_descriptor<D>) and
      (not std::is_base_of_v<Dimensions, D>), int> = 0>
#endif
    constexpr Dimensions& operator=(const D& d)
    {
      if constexpr (not euclidean_vector_space_descriptor<D>)
        if (not get_vector_space_descriptor_is_euclidean(d))
          throw std::invalid_argument{"Argument of dynamic 'Dimensions' assignment operator must be a euclidean vector space descriptor."};
      runtime_size = get_dimension_size_of(d);
      return *this;
    }


#ifdef __cpp_concepts
    template<std::integral Int>
#else
    template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
#endif
    constexpr operator Int()
    {
      return runtime_size;
    }

  protected:

    std::size_t runtime_size;

    friend struct interface::vector_space_traits<Dimensions<dynamic_size>>;
  };


  // ------------------ //
  //  deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<static_vector_space_descriptor D> requires euclidean_vector_space_descriptor<D>
#else
  template<typename D, std::enable_if_t<static_vector_space_descriptor<D> and euclidean_vector_space_descriptor<D>, int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<dimension_size_of_v<D>>;


#ifdef __cpp_concepts
  template<dynamic_vector_space_descriptor D> requires euclidean_vector_space_descriptor<D>
#else
  template<typename D, std::enable_if_t<dynamic_vector_space_descriptor<D> and euclidean_vector_space_descriptor<D>, int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<dynamic_size>;


  explicit Dimensions(const std::size_t&) -> Dimensions<dynamic_size>;


  // ------ //
  //  Axis  //
  // ------ //

  /**
   * \brief Alias for a 1D euclidean \ref vector_space_descriptor object.
   */
  using Axis = Dimensions<1>;


} // OpenKalman::descriptors


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for Dimensions.
   */
  template<std::size_t N>
  struct vector_space_traits<descriptor::Dimensions<N>>
  {
  private:

    using T = descriptor::Dimensions<N>;

  public:

    static constexpr auto
    size(const T& t)
    {
      if constexpr (N == dynamic_size) return t.runtime_size;
      else return std::integral_constant<std::size_t, N>{};
    };


    static constexpr auto
    euclidean_size(const T& t) { return size(t); };


    static constexpr auto
    component_count(const T& t) { return size(t); }


    static constexpr auto
    is_euclidean(const T&) { return std::true_type{}; }


    static constexpr auto
    canonical_equivalent(const T& t)
    {
      if constexpr (N == 0)
        return descriptor::StaticDescriptor<>{};
      else
        return descriptor::Dimensions{t};
    };

  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_DIMENSIONS_HPP
