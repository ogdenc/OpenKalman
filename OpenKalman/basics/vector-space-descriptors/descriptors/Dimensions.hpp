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
 * \brief Definition of the Dimensions class.
 */

#include <cstddef>

#ifndef OPENKALMAN_DIMENSIONS_HPP
#define OPENKALMAN_DIMENSIONS_HPP

namespace OpenKalman
{
  /**
   * \brief A structure representing the dimensions associated with of a particular index.
   * \details The dimension may or may not be known at compile time. If unknown at compile time, the size is set
   * at the time of construction and cannot be modified thereafter.
   * \tparam size The dimension (or <code>dynamic_size</code>, if not known at compile time)
   */
  template<std::size_t size = dynamic_size>
  struct Dimensions;


  // ------------ //
  //  fixed case  //
  // ------------ //

  /**
   * \brief The dimension or size associated with a given index known at compile time.
   * \tparam N The dimension as known at compile time
   */
  template<std::size_t N>
  struct Dimensions
  {
    constexpr Dimensions() = default;

    /// Constructor, taking a \ref fixed_vector_space_descriptor.
#ifdef __cpp_concepts
    template<fixed_vector_space_descriptor D> requires euclidean_vector_space_descriptor<D> and
      (not std::same_as<std::decay_t<D>, Dimensions>) and (dimension_size_of_v<D> == N)
#else
    template<typename D, std::enable_if_t<fixed_vector_space_descriptor<D> and euclidean_vector_space_descriptor<D> and
      not std::is_same_v<std::decay_t<D>, Dimensions> and dimension_size_of<D>::value == N, int> = 0>
#endif
    explicit constexpr Dimensions(D&& d)
    {}

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

    friend struct interface::fixed_vector_space_descriptor_traits<Dimensions<N>>;
  };


  // -------------- //
  //  dynamic case  //
  // -------------- //

  /**
   * \brief The dimension or size associated with a given dynamic index known only at runtime.
   */
  template<>
  struct Dimensions<dynamic_size>
  {
    /// Constructor, taking a \ref fixed_vector_space_descriptor.
#ifdef __cpp_concepts
    template<fixed_vector_space_descriptor D> requires euclidean_vector_space_descriptor<D>
#else
    template<typename D, std::enable_if_t<fixed_vector_space_descriptor<D> and euclidean_vector_space_descriptor<D>, int> = 0>
#endif
    explicit constexpr Dimensions(D&&) : runtime_size {dimension_size_of_v<D>}
    {}


    /// Constructor, taking a \ref dynamic_vector_space_descriptor.
#ifdef __cpp_concepts
    template<dynamic_vector_space_descriptor D> requires (not std::same_as<std::decay_t<D>, Dimensions>)
#else
    template<typename D, std::enable_if_t<dynamic_vector_space_descriptor<D> and not std::is_same_v<std::decay_t<D>, Dimensions>, int> = 0>
#endif
    explicit constexpr Dimensions(D&& d)
      : runtime_size {interface::dynamic_vector_space_descriptor_traits<std::decay_t<D>>{d}.get_size()}
    {
      if constexpr (not euclidean_vector_space_descriptor<D>)
      {
        if (not interface::dynamic_vector_space_descriptor_traits<std::decay_t<D>>{d}.is_euclidean())
          throw std::invalid_argument{"Argument of 'Dimensions' constructor must be a euclidean vector space descriptor."};
      }
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

    Dimensions() = delete;

    std::size_t runtime_size;

    friend struct interface::dynamic_vector_space_descriptor_traits<Dimensions<dynamic_size>>;
  };


  // ------------------ //
  //  deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor D> requires euclidean_vector_space_descriptor<D>
#else
  template<typename D, std::enable_if_t<fixed_vector_space_descriptor<D> and euclidean_vector_space_descriptor<D>, int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<dimension_size_of_v<D>>;


#ifdef __cpp_concepts
  template<dynamic_vector_space_descriptor D>
#else
  template<typename D, std::enable_if_t<dynamic_vector_space_descriptor<D>, int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<dynamic_size>;


  // ------ //
  //  Axis  //
  // ------ //

  /**
   * \brief Alias for a 1D euclidean \ref vector_space_descriptor object.
   */
  using Axis = Dimensions<1>;


  // -------- //
  //  traits  //
  // -------- //

  namespace interface
  {
    /**
     * \internal
     * \brief traits for fixed Dimensions.
     */
#ifdef __cpp_concepts
    template<std::size_t N> requires (N != dynamic_size)
    struct fixed_vector_space_descriptor_traits<Dimensions<N>>
#else
    template<std::size_t N>
    struct fixed_vector_space_descriptor_traits<Dimensions<N>, std::enable_if_t<N != dynamic_size>>
#endif
      : fixed_vector_space_descriptor_traits<std::integral_constant<std::size_t, N>>
    {
      using difference_type = Dimensions<N>;

      static constexpr bool operations_defined = true;
    };


    /**
     * \internal
     * \brief traits for dynamic Dimensions.
     */
    template<>
    struct dynamic_vector_space_descriptor_traits<Dimensions<dynamic_size>> : dynamic_vector_space_descriptor_traits<std::size_t>
    {
    private:

      using Base = dynamic_vector_space_descriptor_traits<std::size_t>;

    public:

      explicit constexpr dynamic_vector_space_descriptor_traits(const Dimensions<dynamic_size>& t) : Base {t.runtime_size} {};
      [[nodiscard]] constexpr std::size_t get_size() const { return Base::get_size(); }
      [[nodiscard]] constexpr std::size_t get_euclidean_size() const { return Base::get_euclidean_size(); }
      [[nodiscard]] constexpr std::size_t get_component_count() const { return Base::get_component_count(); }

      static constexpr bool operations_defined = true;
    };

  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_DIMENSIONS_HPP
