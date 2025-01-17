/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of \ref descriptor::internal::Reverse.
 */

#ifndef OPENKALMAN_DESCRIPTOR_REVERSE_HPP
#define OPENKALMAN_DESCRIPTOR_REVERSE_HPP

#include "basics/utils.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_collection_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_is_euclidean.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_type_index.hpp"
#include "linear-algebra/vector-space-descriptors/functions/to_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/from_euclidean_element.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_wrapped_component.hpp"
#include "linear-algebra/vector-space-descriptors/functions/set_wrapped_component.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief A type representing the reverse of any other \ref vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor C>
#else
  template<typename C>
#endif
  struct Reverse
  {
    /// Default constructor
#ifdef __cpp_concepts
    constexpr Reverse() requires static_vector_space_descriptor<C> = default;
#else
    template<typename mC = C, std::enable_if_t<static_vector_space_descriptor<mC>, int> = 0>
    constexpr Reverse() = default;
#endif


    /**
     * \brief Construct from a \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template <vector_space_descriptor Arg> requires std::constructible_from<C, Arg&&>
#else
    template<typename Arg, std::enable_if_t<vector_space_descriptor<Arg>, std::is_constructible_v<C, Arg&&>, int> = 0>
#endif
    explicit constexpr
    Reverse(Arg&& arg) : my_coordinates {std::forward<Arg>(arg)} {}

  private:

    C my_coordinates;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::vector_space_traits;

  };


  template<typename Arg>
  Reverse(Arg&&) -> Reverse<Arg>;


#ifndef __cpp_lib_ranges
  template<typename Range>
  struct ReverseRange
  {
    constexpr ReverseRange(const ReverseRange& other) = default;
    constexpr ReverseRange(ReverseRange&& other) = default;
    constexpr explicit ReverseRange(const Range& r) : my_range {r} {}
    constexpr auto begin() { using std::begin; return std::make_reverse_iterator(begin(my_range)); }
    constexpr auto begin() const { using std::begin; return std::make_reverse_iterator(begin(my_range)); }
    constexpr auto end() { using std::end; return std::make_reverse_iterator(end(my_range)); }
    constexpr auto end() const { using std::end; return std::make_reverse_iterator(end(my_range)); }
    [[nodiscard]] constexpr std::size_t size() const { using std::size; return size(my_range); }
  private:
    const Range& my_range;
  };
#endif

} // namespace OpenKalman::descriptor::internal


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for descriptor::internal::Reverse.
   */
  template<typename C>
  struct vector_space_traits<descriptor::internal::Reverse<C>>
  {
  private:

    using T = descriptor::internal::Reverse<C>;

  public:

    using scalar_type = descriptor::scalar_type_of_t<C>;


    static constexpr auto
    size(const T& t) { return descriptor::get_dimension_size_of(t.my_coordinates); }


    static constexpr auto
    euclidean_size(const T& t) { return descriptor::get_euclidean_dimension_size_of(t.my_coordinates); }


    static constexpr auto
    collection(const T& t)
    {
      if constexpr (descriptor::static_vector_space_descriptor<C>)
      {
        return internal::tuple_reverse(descriptor::get_collection_of(t.my_coordinates));
      }
      else
      {
#ifdef __cpp_lib_ranges
        return std::ranges::reverse_view {descriptor::get_collection_of(t.my_coordinates)};
#else
        return descriptor::internal::ReverseRange {descriptor::get_collection_of(t.my_coordinates)};
#endif
      }
    }


    static constexpr auto
    is_euclidean(const T& t) { return descriptor::get_vector_space_descriptor_is_euclidean(t.my_coordinates); }


    static constexpr auto
    type_index(const T& t)
    {
      if constexpr (descriptor::static_vector_space_descriptor<C>)
      {
        if constexpr (descriptor::vector_space_component_count_v<C> == 1)
          return descriptor::get_type_index(std::get<0>(descriptor::get_collection_of(t.my_coordinates)));
        else return std::type_index{typeid(T)};
      }
      else return std::type_index{typeid(T)};
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type> and value::index<L> and value::index<S>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
    {
      return descriptor::to_euclidean_element(t.my_coordinates, g, euclidean_size(t) - euclidean_local_index - 1_uz, start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index, const S& euclidean_start)
#endif
    {
      return descriptor::from_euclidean_element(t.my_coordinates, g, size(t) - local_index - 1_uz, euclidean_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> value::value; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      value::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index, const S& start)
#endif
    {
      return descriptor::get_wrapped_component(t.my_coordinates, g, size(t) - local_index - 1_uz, start);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const value::value auto& x,
      const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename X, typename L, typename S, std::enable_if_t<
      value::value<X> and value::index<L> and value::index<S> and
      std::is_invocable<const Setter&, const X&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index, const S& start)
#endif
    {
      return descriptor::set_wrapped_component(t.my_coordinates, s, g, x, size(t) - local_index - 1_uz, start);
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_DESCRIPTOR_REVERSE_HPP
