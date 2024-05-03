/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref fixed_vector_space_descriptor_traits.
 */

#ifndef OPENKALMAN_FIXED_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP
#define OPENKALMAN_FIXED_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP


namespace OpenKalman::interface
{
  /**
   * \brief Traits for a \ref fixed_vector_space_descriptor.
   * \details The traits must define all the members as indicated here.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  struct fixed_vector_space_descriptor_traits;
#else
  struct fixed_vector_space_descriptor_traits
  {
    /// \brief The number of dimensions at compile time.
    static constexpr std::size_t size = 0;

    /// \brief The number of dimensions after transforming to Euclidean space.
    static constexpr std::size_t euclidean_size = 0;

    /// \brief The number of atomic component parts at compile time.
    static constexpr std::size_t component_count = 0;

    /// \brief The type of the \ref vector_space_descriptor when tensors having respective vector_space_descriptor T are subtracted.
    /// \details For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Dimensions<1>.
    /// So if <code>T</code> is Distance, the resulting <code>difference_type</code> will be Dimensions<1>.
    using difference_type = std::decay_t<T>;

    /// \brief Whether the \ref vector_space_descriptor object is known at compile time to describe Euclidean coordinates (and in this case, size == euclidean_size).
    static constexpr bool always_euclidean = false;

    /// \brief Whether arithmetic operations (e.g., addition, subtraction) are defined for this \ref vector_space_descriptor object.
    static constexpr bool operations_defined = false;


    /**
     * \brief Maps an element from coordinates in modular space to coordinates in Euclidean space.
     * \note For \ref fixed_vector_space_descriptor, this must be a static function.
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param euclidean_local_index A local index accessing the coordinate in Euclidean space
     * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr scalar_type auto to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; } = delete;
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start) = delete;
#endif


    /**
     * \brief The inverse of <code>to_euclidean_element</code>. Maps coordinates in Euclidean space back into modular space.
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the coordinate in modular space.
     * \param euclidean_start The starting location in Euclidean space within any larger set of \ref vector_space_descriptor
     */
#ifdef __cpp_concepts
    static constexpr scalar_type auto
    from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; } = delete;
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start) = delete;
#endif


    /**
     * \brief Gets an element from a matrix or tensor object and wraps the result.
     * \details The wrapping operation is equivalent to mapping from modular space to Euclidean space and then back again,
     * or in other words, performing <code>to_euclidean_element</code> followed by <code>from_euclidean_element<code>.
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param local_index A local index accessing the element.
     * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    static constexpr scalar_type auto get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; } = delete;
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start) = delete;
#endif


    /**
     * \brief Set an angle and then wrapping.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \param s An element setter that sets an element at the location of index i (e.g., <code>std::function&lt;void(std::size_t, double)&rt;</code>)
     * \param g An element getter mapping an index i of type std::size_t to an element of \ref scalar_type
     * (e.g., <code>std::function&lt;double(std::size_t)&rt;</code>)
     * \param x The new value to be set.
     * \param local_index A local index accessing the element.
     * \param start The starting location of the element within any larger set of \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    static constexpr void set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; } = delete;
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start) = delete;
#endif

  };
#endif //DOXYGEN_SHOULD_SKIP_THIS


  // ------------------------------- //
  //  definition for integral types  //
  // ------------------------------- //

  /**
   * \internal
   * \brief traits for a \ref static_index_value.
   */
#ifdef __cpp_concepts
  template<static_index_value T>
  struct fixed_vector_space_descriptor_traits<T>
#else
  template<typename T>
  struct fixed_vector_space_descriptor_traits<T, std::enable_if_t<static_index_value<T>>>
#endif
  {
    static constexpr std::size_t size = static_cast<std::size_t>(T{});
    static constexpr std::size_t euclidean_size = size;
    static constexpr std::size_t component_count = size;
    using difference_type = T;
    static constexpr bool always_euclidean = true;


#ifdef __cpp_concepts
    static constexpr scalar_type auto to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start)
#endif
    {
      return g(start + euclidean_local_index);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start)
#endif
    {
      return g(euclidean_start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr scalar_type auto get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ {g(i)} -> scalar_type; }
#else
    template<typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type>, int> = 0>
    static constexpr auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start)
#endif
    {
      return g(start + local_index);
    }


#ifdef __cpp_concepts
    static constexpr void set_wrapped_component(const auto& s, const auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start)
    requires requires (std::size_t i){ s(x, i); {x} -> scalar_type; }
#else
    template<typename S, typename G, std::enable_if_t<scalar_type<typename std::invoke_result<G, std::size_t>::type> and
      std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start)
#endif
    {
      s(x, start + local_index);
    }

  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_FIXED_VECTOR_SPACE_DESCRIPTOR_TRAITS_HPP
