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
 * \brief Definition of \ref descriptor::internal::Concatenate.
 */

#ifndef OPENKALMAN_DESCRIPTOR_CONCATENATE_HPP
#define OPENKALMAN_DESCRIPTOR_CONCATENATE_HPP

#include <type_traits>
#include <array>
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
#include "linear-algebra/vector-space-descriptors/traits/scalar_type_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/internal/AnyAtomicVectorSpaceDescriptor.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief A type representing the concatenation of two other \ref vector_space_descriptor objects.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor C1, vector_space_descriptor C2>
#else
  template<typename C1, typename C2>
#endif
  struct Concatenate
  {
    /// Default constructor
#ifdef __cpp_concepts
    constexpr Concatenate() requires static_vector_space_descriptor<C1> and static_vector_space_descriptor<C2> = default;
#else
    template<typename mC1 = C1, typename mC2 = C2, std::enable_if_t<
      static_vector_space_descriptor<mC1> and static_vector_space_descriptor<mC2>, int> = 0>
    constexpr Concatenate() {};
#endif


    /**
     * \brief Construct from two \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    template <vector_space_descriptor Arg1, vector_space_descriptor Arg2> requires
      std::constructible_from<C1, Arg1&&> and std::constructible_from<C2, Arg2&&>
#else
    template<typename Arg1, typename Arg2, std::enable_if_t<vector_space_descriptor<Arg1> and vector_space_descriptor<Arg2> and
      std::is_constructible_v<C1, Arg1&&> and std::is_constructible_v<C2, Arg2&&>, int> = 0>
#endif
    explicit constexpr
    Concatenate(Arg1&& arg1, Arg2&& arg2)
      : my_coordinates1 {std::forward<Arg1>(arg1)}, my_coordinates2 {std::forward<Arg2>(arg2)} {}

  private:

    C1 my_coordinates1;
    C2 my_coordinates2;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::vector_space_traits;

  };


  template<typename Arg1, typename Arg2>
  Concatenate(Arg1&&, Arg2&&) -> Concatenate<
    std::conditional_t<static_vector_space_descriptor<Arg1>, std::decay_t<Arg1>, Arg1>,
    std::conditional_t<static_vector_space_descriptor<Arg2>, std::decay_t<Arg2>, Arg2>>;


#if __cpp_lib_containers_ranges < 202202L or __cpp_lib_ranges_concat < 202403L
  template<typename Range1, typename Range2>
  struct ConcatenateRange
  {
    struct Iterator
    {
      using iterator_category = std::random_access_iterator_tag;

      using difference_type = std::ptrdiff_t;

      using value_type = AnyAtomicVectorSpaceDescriptor<std::common_type_t<scalar_type_of_t<Range1>, scalar_type_of_t<Range2>>>;

      template<typename R1, typename R2>
      constexpr Iterator(R1* r1, R2* r2, const std::size_t p) : my_range1{r1}, my_range2{r2}, pos{p} {}

      constexpr Iterator() = default;
      constexpr Iterator(const Iterator& other) = default;
      constexpr Iterator(Iterator&& other) noexcept = default;

      constexpr Iterator& operator=(const Iterator& other) = default;
      constexpr Iterator& operator=(Iterator&& other) noexcept = default;

      constexpr value_type operator*() const
      {
        using std::size;
        auto s1 = size(*my_range1);
        if (pos < s1) return static_cast<value_type>((*my_range1)[pos]);
        else return static_cast<value_type>((*my_range2)[pos - s1]);
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
        return Iterator {my_range1, my_range2, pos + diff};
      }

      friend constexpr auto operator+(const difference_type diff, const Iterator& it) noexcept {
        return Iterator {it.my_range1, it.my_range2, diff + it.pos};
      }

      constexpr auto operator-(const difference_type diff) const noexcept
      {
        return Iterator {my_range1, my_range2, pos - diff};
      }

      constexpr auto operator+(const Iterator& other) const noexcept
      {
        return Iterator {my_range1, my_range2, pos + other.pos};
      }

      constexpr difference_type operator-(const Iterator& other) const noexcept
      {
        return pos - other.pos;
      }

      constexpr value_type operator[](difference_type offset) const
      {
        using std::size;
        auto s1 = size(*my_range1);
        auto p = pos + offset;
        if (p < s1) return static_cast<value_type>((*my_range1)[p]);
        else return static_cast<value_type>((*my_range2)[p - s1]);
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

      const std::remove_reference_t<Range1> * my_range1;

      const std::remove_reference_t<Range2> * my_range2;

      std::size_t pos;

    }; // struct Iterator


    constexpr ConcatenateRange(Range1 r1, Range2 r2) : my_range1 {r1}, my_range2 {r2} {}

    [[nodiscard]] constexpr std::size_t size() const
    {
      using std::size; return size(my_range1) + size(my_range2);
    }

    constexpr auto begin()
    {
      return Iterator {std::addressof(my_range1), std::addressof(my_range2), 0};
    }

    constexpr auto begin() const
    {
      return Iterator {std::addressof(my_range1), std::addressof(my_range2), 0};
    }

    constexpr auto end()
    {
      return Iterator {std::addressof(my_range1), std::addressof(my_range2), size()};
    }

    constexpr auto end() const
    {
      return Iterator {std::addressof(my_range1), std::addressof(my_range2), size()};
    }

  private:

    Range1 my_range1;

    Range2 my_range2;

  }; // struct ConcatenateRange


  template<typename Arg1, typename Arg2>
  ConcatenateRange(Arg1&&, Arg2&&) -> ConcatenateRange<Arg1, Arg2>;
#endif

} // namespace OpenKalman::descriptor::internal


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for descriptor::internal::Reverse.
   */
  template<typename C1, typename C2>
  struct vector_space_traits<descriptor::internal::Concatenate<C1, C2>>
  {
  private:

    using T = descriptor::internal::Concatenate<C1, C2>;

  public:

    using scalar_type = std::conditional_t<
      descriptor::dynamic_vector_space_descriptor<C1> and descriptor::dynamic_vector_space_descriptor<C2>,
      descriptor::scalar_type_of_t<std::common_type_t<descriptor::scalar_type_of_t<C1>, descriptor::scalar_type_of_t<C2>>>,
      std::conditional_t<descriptor::dynamic_vector_space_descriptor<C1>, descriptor::scalar_type_of_t<C1>, descriptor::scalar_type_of_t<C2>>>;


    static constexpr auto
    size(const T& t)
    {
      return value::operation {std::plus<>{},
        descriptor::get_dimension_size_of(t.my_coordinates1),
        descriptor::get_dimension_size_of(t.my_coordinates2)};
    }


    static constexpr auto
    euclidean_size(const T& t)
    {
      return value::operation {std::plus<>{},
        descriptor::get_euclidean_dimension_size_of(t.my_coordinates1),
        descriptor::get_euclidean_dimension_size_of(t.my_coordinates2)};
    }

  private:

    template<typename...Ts>
    static constexpr auto combine_euclidean()
    {
      constexpr std::size_t N = (0 + ... + descriptor::dimension_size_of_v<Ts>);
      if constexpr (N == 0) return std::tuple {};
      else return std::array {descriptor::Dimensions<N>{}};
    }


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
      if constexpr (descriptor::static_vector_space_descriptor<C1> and descriptor::static_vector_space_descriptor<C2>)
      {
        if constexpr (descriptor::euclidean_vector_space_descriptor<C1> and descriptor::euclidean_vector_space_descriptor<C2>)
        {
          return combine_euclidean<C1, C2>();
        }
        else if constexpr (descriptor::dimension_size_of_v<C1> == 0)
        {
          return descriptor::get_collection_of(C2{});
        }
        else if constexpr (descriptor::dimension_size_of_v<C2> == 0)
        {
          return descriptor::get_collection_of(C1{});
        }
        else
        {
          auto c1 = descriptor::get_collection_of(C1{});
          constexpr std::size_t s1 = std::tuple_size_v<std::decay_t<decltype(c1)>> - 1;
          using c1_last = std::tuple_element_t<s1, std::decay_t<decltype(c1)>>;
          auto c2 = descriptor::get_collection_of(C2{});
          constexpr std::size_t s2 = std::tuple_size_v<std::decay_t<decltype(c2)>>;
          using c2_first = std::tuple_element_t<0, std::decay_t<decltype(c2)>>;
          if constexpr (descriptor::euclidean_vector_space_descriptor<c1_last> and descriptor::euclidean_vector_space_descriptor<c2_first>)
          {
            return std::tuple_cat(
              internal::tuple_slice<0, s1>(std::move(c1)),
              combine_euclidean<c1_last, c2_first>(),
              internal::tuple_slice<1, s2>(std::move(c2)));
          }
          else
          {
            return std::tuple_cat(std::move(c1), std::move(c2));
          }
        }
      }
      else
      {
#if __cpp_lib_containers_ranges >= 202202L and __cpp_lib_ranges_concat >= 202403L
        return std::views::concat(
          tuple_to_range(descriptor::get_collection_of(std::forward<Arg>(arg).my_coordinates1)),
          tuple_to_range(descriptor::get_collection_of(std::forward<Arg>(arg).my_coordinates2)));
#else
        return descriptor::internal::ConcatenateRange {
          tuple_to_range(descriptor::get_collection_of(std::forward<Arg>(arg).my_coordinates1)),
          tuple_to_range(descriptor::get_collection_of(std::forward<Arg>(arg).my_coordinates2))};
#endif
      }
    }


    static constexpr auto
    is_euclidean(const T& t)
    {
      return value::operation {std::logical_and<>{},
        descriptor::get_vector_space_descriptor_is_euclidean(t.my_coordinates1),
        descriptor::get_vector_space_descriptor_is_euclidean(t.my_coordinates2)};
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
      auto es1 = descriptor::get_euclidean_dimension_size_of(t.my_coordinates1);
      if (euclidean_local_index < es1) return descriptor::to_euclidean_element(t.my_coordinates1, g, euclidean_local_index, start);
      auto s1 = descriptor::get_dimension_size_of(t.my_coordinates1);
      return descriptor::to_euclidean_element(t.my_coordinates2, g, euclidean_local_index - es1, start + s1);
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
      auto s1 = descriptor::get_dimension_size_of(t.my_coordinates1);
      if (local_index < s1) return descriptor::from_euclidean_element(t.my_coordinates1, g, local_index, euclidean_start);
      auto es1 = descriptor::get_euclidean_dimension_size_of(t.my_coordinates1);
      return descriptor::from_euclidean_element(t.my_coordinates2, g, local_index - s1, euclidean_start + es1);
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
      auto s1 = descriptor::get_dimension_size_of(t.my_coordinates1);
      if (local_index < s1) return descriptor::get_wrapped_component(t.my_coordinates1, g, local_index, start);
      return descriptor::get_wrapped_component(t.my_coordinates2, g, local_index - s1, start + s1);
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
      auto s1 = descriptor::get_dimension_size_of(t.my_coordinates1);
      if (local_index < s1) return descriptor::set_wrapped_component(t.my_coordinates1, s, g, x, local_index, start);
      return descriptor::set_wrapped_component(t.my_coordinates2, s, g, x, local_index - s1, start + s1);
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_DESCRIPTOR_CONCATENATE_HPP
