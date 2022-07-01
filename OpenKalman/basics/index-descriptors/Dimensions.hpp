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


  /**
   * \overload
   * \brief The dimension or size associated with a given index known at compile time.
   * \tparam N The dimension as known at compile time
   */
  template<std::size_t N>
  struct Dimensions
  {
    /// \returns The number of dimensions
    static constexpr std::size_t size() { return N; }


    constexpr Dimensions() = default;


    /// Constructor, taking a fixed-size index descriptor.
#ifdef __cpp_concepts
    template<fixed_index_descriptor D> requires euclidean_index_descriptor<D> and
      (not std::same_as<std::decay_t<D>, Dimensions>) and (dimension_size_of_v<D> == N)
#else
    template<typename D, std::enable_if_t<fixed_index_descriptor<D> and euclidean_index_descriptor<D> and
      not std::is_same_v<std::decay_t<D>, Dimensions> and dimension_size_of<D>::value == N, int> = 0>
#endif
    explicit constexpr Dimensions(D&& d)
    {}


    /**
     * \brief Maps an element to coordinates in Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index relative to the Euclidean-transformed coordinates (starting at 0)
     * \param start The starting index within the index descriptor
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar>
#else
    template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
    static constexpr auto
    to_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t euclidean_local_index, std::size_t start)
    {
      return g(start + euclidean_local_index);
    }


    /**
     * \brief Maps a coordinate in Euclidean space to an element.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index relative to the original coordinates (starting at 0)
     * \param start The starting index within the Euclidean-transformed indices
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar>
#else
    template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
    static constexpr auto
    from_euclidean_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t euclidean_start)
    {
      return g(euclidean_start + local_index);
    }


    /**
     * \brief Perform modular wrapping of an element.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar>
#else
    template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
    static constexpr auto
    wrap_get_element(const std::function<Scalar(std::size_t)>& g, std::size_t local_index, std::size_t start)
    {
      return g(start + local_index);
    }


    /**
     * \brief Set an element and then perform any necessary modular wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \tparam Scalar The scalar type (e.g., double).
     * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
     * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the angle (in this case, it must be 0)
     * \param start The starting location of the angle within any larger set of index type descriptors
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar>
#else
    template<typename Scalar, std::enable_if_t<scalar_type<Scalar>, int> = 0>
#endif
    static constexpr void
    wrap_set_element(const std::function<void(Scalar, std::size_t)>& s, const std::function<Scalar(std::size_t)>& g,
            Scalar x, std::size_t local_index, std::size_t start)
    {
      s(x, start + local_index);
    }

  };


  /**
   * \overload
   * \brief The dimension or size associated with a given dynamic index known only at runtime.
   */
  template<>
  struct Dimensions<dynamic_size>
  {
  protected:

    std::size_t runtime_size;

  public:

    Dimensions() = delete;


    /// Constructor, taking a dynamic index descriptor.
#ifdef __cpp_concepts
    template<euclidean_index_descriptor D> requires (not std::same_as<std::decay_t<D>, Dimensions>)
#else
    template<typename D, std::enable_if_t<euclidean_index_descriptor<D> and
      not std::is_same_v<std::decay_t<D>, Dimensions>, int> = 0>
#endif
    explicit constexpr Dimensions(D&& d)
      : runtime_size {interface::IndexDescriptorSize<std::decay_t<D>>::get(std::forward<D>(d))}
    {}


    /// \returns The number of dimensions
    constexpr std::size_t size() const { return runtime_size; }

  };


#ifdef __cpp_concepts
  template<fixed_index_descriptor D> requires euclidean_index_descriptor<D>
#else
  template<typename D, std::enable_if_t<fixed_index_descriptor<D> and euclidean_index_descriptor<D>, int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<dimension_size_of_v<D>>;


#ifdef __cpp_concepts
  template<euclidean_index_descriptor D> requires (not fixed_index_descriptor<D>)
#else
  template<typename D, std::enable_if_t<euclidean_index_descriptor<D> and (not fixed_index_descriptor<D>), int> = 0>
#endif
  explicit Dimensions(D&&) -> Dimensions<dynamic_size>;


  using Axis = Dimensions<1>;


  namespace interface
  {
    /**
     * \internal
     * \brief Axis is represented by one coordinate.
     */
    template<std::size_t size>
    struct IndexDescriptorSize<Dimensions<size>> : std::integral_constant<std::size_t, size>
    {
      static constexpr std::size_t get(const Dimensions<size>& t) { return t.size(); }
    };


    /**
     * \internal
     * \brief Axis is represented by one coordinate in Euclidean space.
     */
    template<std::size_t size>
    struct EuclideanIndexDescriptorSize<Dimensions<size>> : IndexDescriptorSize<Dimensions<size>> {};


    /**
     * \internal
     * \brief The number of atomic components.
     */
    template<std::size_t size>
    struct IndexDescriptorComponentCount<Dimensions<size>> : IndexDescriptorSize<Dimensions<size>> {};


    /**
     * \internal
     * \brief The type of the result when subtracting two Axis values.
     * \details A difference between two Dimensions objects is also of type Dimensions.
     */
    template<std::size_t size>
    struct IndexDescriptorDifferenceType<Dimensions<size>> { using type = Dimensions<size>; };


    /**
     * \internal
     * \brief Dimensions is untyped.
     */
    template<std::size_t size>
    struct IndexDescriptorIsUntyped<Dimensions<size>> : std::true_type
    {
      constexpr static std::size_t get(const Dimensions<size>& t) { return true; }
    };

  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_DIMENSIONS_HPP
