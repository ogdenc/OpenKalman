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
 * \brief Definition of \ref descriptor::internal::Replicate.
 */

#ifndef OPENKALMAN_DESCRIPTOR_REPLICATE_HPP
#define OPENKALMAN_DESCRIPTOR_REPLICATE_HPP

#include <type_traits>
#include <array>

#include "basics/utils.hpp"
#include "linear-algebra/values/concepts/index.hpp"
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
#include "linear-algebra/vector-space-descriptors/traits/scalar_type_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/internal/AnyAtomicVectorSpaceDescriptor.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief A type representing the replication of other \ref vector_space_descriptor objects.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor C, value::index N>
#else
  template<typename C, typename N>
#endif
  struct Replicate
  {
    /// Default constructor
#ifdef __cpp_concepts
    constexpr Replicate() requires static_vector_space_descriptor<C> and value::fixed<N> = default;
#else
    template<typename mC = C, typename mN = N, std::enable_if_t<
      static_vector_space_descriptor<mC> and value::fixed<mN>, int> = 0>
    constexpr Replicate() {};
#endif


    /**
     * \brief Construct from a \ref vector_space_descriptor and a \ref value::index.
     */
#ifdef __cpp_concepts
    template <vector_space_descriptor Arg, value::index I> requires
      std::constructible_from<C, Arg&&> and std::constructible_from<N, I&&>
#else
    template<typename Arg, typename I, std::enable_if_t<vector_space_descriptor<Arg> and vvalie::index<I> and
      std::is_constructible_v<C, Arg&&> and std::is_constructible_v<N, I&&>, int> = 0>
#endif
    explicit constexpr
    Replicate(Arg&& arg, I&& i)
      : my_coordinates {std::forward<Arg>(arg)}, my_index {std::forward<I>(i)} {}

  private:

    C my_coordinates;
    N my_index;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::vector_space_traits;

  };


  template<typename Arg, typename I>
  Replicate(Arg&&, const I&) -> Replicate<std::conditional_t<static_vector_space_descriptor<Arg>, std::decay_t<Arg>, Arg>, I>;


  template<typename Range>
  struct ReplicateRange
  {
    struct Iterator
    {
      using iterator_category = std::random_access_iterator_tag;

      using difference_type = std::ptrdiff_t;

      using value_type = AnyAtomicVectorSpaceDescriptor<scalar_type_of_t<Range>>;

      template<typename R>
      constexpr Iterator(R* r, const std::size_t p) : my_range{r}, pos{p} {}

      constexpr Iterator() = default;
      constexpr Iterator(const Iterator& other) = default;
      constexpr Iterator(Iterator&& other) noexcept = default;

      constexpr Iterator& operator=(const Iterator& other) = default;
      constexpr Iterator& operator=(Iterator&& other) noexcept = default;

      constexpr value_type operator*() const
      {
        using std::size;
        return (*my_range)[pos % size(*my_range)];
      }

      constexpr auto& operator++() noexcept
      {
        ++pos; return *this;
      }

      constexpr auto operator++(int) noexcept
      {
        auto temp = *this; ++*this; return temp;
      }

      constexpr auto& operator--() noexcept
      {
        --pos; return *this;
      }

      constexpr auto operator--(int) noexcept
      {
        auto temp = *this; --*this; return temp;
      }

      constexpr auto& operator+=(const difference_type diff) noexcept
      {
        pos += diff; return *this;
      }

      constexpr auto& operator-=(const difference_type diff) noexcept
      {
        pos -= diff; return *this;
      }

      constexpr auto operator+(const difference_type diff) const noexcept
      {
        return Iterator {my_range, pos + diff};
      }

      friend constexpr auto operator+(const difference_type diff, const Iterator& it) noexcept {
        return Iterator {it.my_range, diff + it.pos};
      }

      constexpr auto operator-(const difference_type diff) const noexcept
      {
        return Iterator {my_range, pos - diff};
      }

      constexpr auto operator+(const Iterator& other) const noexcept
      {
        return Iterator {my_range, pos + other.pos};
      }

      constexpr difference_type operator-(const Iterator& other) const noexcept
      {
        return pos - other.pos;
      }

      constexpr value_type operator[](difference_type offset) const
      {
        using std::size;
        return (*my_range)[(pos + offset) % size(*my_range)];
      }

      constexpr bool operator==(const Iterator& other) const noexcept { return pos == other.pos; }

#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const Iterator& other) const noexcept { return pos <=> other.pos; }
#else
      constexpr bool operator!=(const Iterator& other) const noexcept { return pos != other.pos; }
      constexpr bool operator<(const Iterator& other) const noexcept { return pos < other.pos; }
      constexpr bool operator>(const Iterator& other) const noexcept { return pos > other.pos; }
      constexpr bool operator<=(const Iterator& other) const noexcept { return pos <= other.pos; }
      constexpr bool operator>=(const Iterator& other) const noexcept { return pos >= other.pos; }
#endif

    private:

      const std::remove_reference_t<Range> * my_range;

      std::size_t pos;

    }; // struct Iterator


    constexpr ReplicateRange(Range r, std::size_t n) : my_range {r}, my_index {n} {}

    [[nodiscard]] constexpr std::size_t size() const
    {
      using std::size; return size(my_range) * my_index;
    }

    constexpr auto begin()
    {
      return Iterator {std::addressof(my_range), 0};
    }

    constexpr auto begin() const
    {
      return Iterator {std::addressof(my_range), 0};
    }

    constexpr auto end()
    {
      return Iterator {std::addressof(my_range), size()};
    }

    constexpr auto end() const
    {
      return Iterator {std::addressof(my_range), size()};
    }

  private:

    Range my_range;

    std::size_t my_index;

  }; // struct ReplicateRange


  template<typename Arg, typename I>
  ReplicateRange(Arg&&, I&&) -> ReplicateRange<Arg>;

} // namespace OpenKalman::descriptor::internal


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for descriptor::internal::Reverse.
   */
  template<typename C, typename N>
  struct vector_space_traits<descriptor::internal::Replicate<C, N>>
  {
  private:

    using T = descriptor::internal::Replicate<C, N>;

  public:

    using scalar_type = descriptor::scalar_type_of_t<C>;


    static constexpr auto
    size(const T& t)
    {
      return value::operation {std::multiplies<>{}, descriptor::get_dimension_size_of(t.my_coordinates), t.my_index};
    }


    static constexpr auto
    euclidean_size(const T& t)
    {
      return value::operation {std::multiplies<>{}, descriptor::get_euclidean_dimension_size_of(t.my_coordinates), t.my_index};
    }

  private:

    template<typename Arg, std::size_t...Ix>
    static constexpr auto
    tuple_to_range_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      return std::array {descriptor::internal::AnyAtomicVectorSpaceDescriptor<scalar_type> {std::get<Ix>(std::forward<Arg>(arg))}...};
    }


    template<typename Arg>
    static constexpr auto
    tuple_to_range(Arg&& arg)
    {
      if constexpr (descriptor::vector_space_descriptor_range<Arg>)
        return std::forward<Arg>(arg);
      else
        return tuple_to_range_impl(std::forward<Arg>(arg), std::make_index_sequence<std::tuple_size_v<std::decay_t<Arg>>>{});
    }

  public:

    template<typename Arg>
    static constexpr auto
    collection(Arg&& arg)
    {
      if constexpr (descriptor::static_vector_space_descriptor<C> and value::fixed<N>)
      {
        if constexpr (descriptor::dimension_size_of_v<C> == 0)
        {
          return std::tuple {};
        }
        else if constexpr (descriptor::euclidean_vector_space_descriptor<C>)
        {
          return std::array {descriptor::Dimensions {size(arg)}};
        }
        else
        {
          constexpr std::size_t n = value::fixed_number_of_v<N>;
          auto coll = descriptor::get_collection_of(C{});
          using Coll =  std::decay_t<decltype(coll)>;
          constexpr std::size_t s = std::tuple_size_v<Coll>;
          using c_first = std::tuple_element_t<0, Coll>;
          using c_last = std::tuple_element_t<s - 1, Coll>;
          if constexpr (descriptor::euclidean_vector_space_descriptor<c_last> and
            descriptor::euclidean_vector_space_descriptor<c_first> and n >= 2)
          {
            auto e = std::tuple {descriptor::Dimensions<descriptor::dimension_size_of_v<c_first> + descriptor::dimension_size_of_v<c_last>>{}};
            return std::tuple_cat(
              internal::tuple_slice<0, s - 1>(coll),
              internal::tuple_flatten(internal::fill_tuple<n - 2>(std::tuple_cat(e, internal::tuple_slice<1, s - 1>(coll)))),
              e, internal::tuple_slice<1, s>(coll));
          }
          else
          {
            return internal::tuple_flatten(internal::fill_tuple<n>(std::move(coll)));
          }
        }
      }
      else
      {
        return descriptor::internal::ReplicateRange {
          tuple_to_range(descriptor::get_collection_of(std::forward<Arg>(arg).my_coordinates)), arg.my_index};
      }
    }


    static constexpr auto
    is_euclidean(const T& t)
    {
      return descriptor::get_vector_space_descriptor_is_euclidean(t.my_coordinates);
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
      std::size_t n = descriptor::get_euclidean_dimension_size_of(t.my_coordinates);
      return descriptor::to_euclidean_element(t.my_coordinates, g, euclidean_local_index % n, start + euclidean_local_index / n);
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
      auto n = descriptor::get_dimension_size_of(t.my_coordinates);
      return descriptor::from_euclidean_element(t.my_coordinates, g, local_index % n, euclidean_start + local_index / n);
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
      auto n = descriptor::get_dimension_size_of(t.my_coordinates);
      return descriptor::get_wrapped_component(t.my_coordinates, g, local_index % n, start + local_index / n);
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
      auto n = descriptor::get_dimension_size_of(t.my_coordinates);
      return descriptor::set_wrapped_component(t.my_coordinates, s, g, x, local_index % n, start + local_index / n);
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_DESCRIPTOR_REPLICATE_HPP
