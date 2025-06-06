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
 * \brief Definition of \ref coordinates::internal::Slice.
 */

#ifndef OPENKALMAN_DESCRIPTOR_SLICE_HPP
#define OPENKALMAN_DESCRIPTOR_SLICE_HPP

#include <type_traits>
#include <array>

#include "values/concepts/index.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/functions/get_dimension.hpp"
#include "linear-algebra/coordinates/functions/get_stat_dimension.hpp"
#include "linear-algebra/coordinates/functions/internal/get_component_collection.hpp"
#include "linear-algebra/coordinates/functions/get_is_euclidean.hpp"
#include "linear-algebra/coordinates/functions/to_stat_space.hpp"
#include "linear-algebra/coordinates/functions/from_stat_space.hpp"
#include "linear-algebra/coordinates/functions/get_wrapped_component.hpp"
#include "linear-algebra/coordinates/functions/set_wrapped_component.hpp"
#include "linear-algebra/coordinates/traits/scalar_type_of_deprecated.hpp"
#include "linear-algebra/coordinates/descriptors/Any.hpp"

namespace OpenKalman::coordinates::internal
{
  /**
   * \internal
   * \brief A type representing a slice of another \ref coordinates::pattern objects.
   */
#ifdef __cpp_concepts
  template<pattern C, values::index Offset, values::index Extent> requires dynamic_pattern<C> or
    ((values::dynamic<Offset> or values::fixed_number_of_v<Offset> <= dimension_of_v<C>) and
    (values::dynamic<Extent> or values::fixed_number_of_v<Extent> <= dimension_of_v<C>) and
    (values::dynamic<Offset> or values::dynamic<Extent> or values::fixed_number_of_v<Offset> + values::fixed_number_of_v<Extent> <= dimension_of_v<C>))
#else
  template<typename C, typename Offset, typename Extent>
#endif
  struct Slice
  {
#ifndef __cpp_concepts
    if constexpr (fixed_pattern<C>)
    {
      if constexpr (values::fixed<Offset>)
        static_assert(values::fixed_number_of_v<Offset> <= dimension_of_v<C>);
      if constexpr (values::fixed<Extent>)
        static_assert(values::fixed_number_of_v<Extent> <= dimension_of_v<C>);
      if constexpr (values::fixed<Offset> and values::fixed<Extent>)
        static_assert(values::fixed_number_of_v<Offset> + values::fixed_number_of_v<Extent> <= dimension_of_v<C>);
    }
#endif


    /// Default constructor
#ifdef __cpp_concepts
    constexpr Slice() requires fixed_pattern<C> and values::fixed<Offset> and values::fixed<Extent> = default;
#else
    template<typename mC = C, typename mO = Offset, typename mE = Extent, std::enable_if_t<
      fixed_pattern<mC> and values::fixed<mO> and values::fixed<mE>, int> = 0>
    constexpr Slice() {};
#endif


    /**
     * \brief Construct from a \ref coordinates::pattern, an offset, and an extent.
     */
#ifdef __cpp_concepts
    template <pattern Arg, values::index an_offset, values::index an_extent> requires
      std::constructible_from<C, Arg&&> and std::constructible_from<Offset, an_offset&&> and
      std::constructible_from<Extent, an_extent&&>
#else
    template<typename Arg, typename an_offset, typename an_extent, std::enable_if_t<pattern<Arg> and
      values::index<an_offset> and values::index<an_extent> and std::is_constructible_v<C, Arg&&> and
      std::is_constructible_v<Offset, an_offset&&> and std::is_constructible_v<Extent, an_extent&&>, int> = 0>
#endif
    explicit constexpr
    Slice(Arg&& arg, an_offset&& offset, an_extent&& extent)
      : my_coordinates {std::forward<Arg>(arg)},
        my_offset {std::forward<an_offset>(offset)},
        my_extent {std::forward<an_extent>(extent)} {}

  private:

    C my_coordinates;
    Offset my_offset;
    Extent my_extent;

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::coordinate_descriptor_traits;

  };


  template<typename Arg, typename an_offset, typename an_extent>
  Slice(Arg&&, const an_offset&, const an_extent&)
    -> Slice<std::conditional_t<fixed_pattern<Arg>, std::decay_t<Arg>, Arg>, an_offset, an_extent>;


  template<typename Range>
  struct SliceRange
  {
#ifdef __cpp_lib_ranges
    namespace ranges = std::ranges;
#endif
    struct Iterator
    {
      using iterator_category = std::random_access_iterator_tag;

      using difference_type = std::ptrdiff_t;

      using value_type = coordinates::Any<scalar_type_of_t<Range>>;

      template<typename R>
      constexpr Iterator(R* r, const std::size_t p) : my_range{r}, pos{p} {}

      constexpr Iterator() = default;
      constexpr Iterator(const Iterator& other) = default;
      constexpr Iterator(Iterator&& other) noexcept = default;

      constexpr Iterator& operator=(const Iterator& other) = default;
      constexpr Iterator& operator=(Iterator&& other) noexcept = default;

      constexpr value_type operator*() const
      {
        return (*my_range)[pos % ranges::size(*my_range)];
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
        return (*my_range)[(pos + offset) % ranges::size(*my_range)];
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


    constexpr SliceRange(Range&& r, std::size_t n) : my_range {r}, my_index {n} {}

    [[nodiscard]] constexpr std::size_t size() const
    {
      return ranges::size(my_range) * my_index;
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

    std::size_t my_offset, my_extent;

  }; // struct ReplicateRange


  template<typename Arg, typename an_offset, typename an_extent>
  SliceRange(Arg&&, an_offset&&, an_extent&&) -> ReplicateRange<Arg>;

} // namespace OpenKalman::coordinates::internal


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for coordinates::internal::Reverse.
   */
  template<typename C, typename Offset, typename Extent>
  struct coordinate_set_traits<coordinates::internal::Slice<C, Offset, Extent>>
  {
  private:

    using T = coordinates::internal::Slice<C, Offset, Extent>;

  public:

    static constexpr bool is_specialized = true;


    using scalar_type = coordinates::scalar_type_of_t<C>;


    static constexpr auto
    size(const T& t)
    {
      return t.my_extent;
    }

  private:

    template<std::size_t position = 0, std::size_t element = 0, typename...Ds, typename Tup>
    auto fixed_slice(const Tup& tup)
    {
      constexpr auto offset = values::fixed_number_of_v<Offset>;
      constexpr auto extent = values::fixed_number_of_v<Extent>;
      if constexpr (position < offset + extent)
      {
        using Element = std::tuple_element_t<element, Tup>;
        constexpr auto next_position = position + coordinates::dimension_of_v<Element>;
        if constexpr (position < offset)
        {
          static_assert(next_position <= offset, "Slice must begin on a coordinate-descriptor boundary");
          return fixed_slice<offset, extent, next_position, element + 1, Ds...>(tup);
        }
        else
        {
          return fixed_slice<offset, extent, next_position, element + 1, Ds..., Element>(tup);
        }
      }
      else
      {
        static_assert(position == offset + extent, "Slice must end on a coordinate-descriptor boundary");
        return std::tuple {Ds{}...};
      }
    };


    template<typename Arg, std::size_t...Ix>
    static constexpr auto
    tuple_to_range_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      return std::array {coordinates::Any<scalar_type> {std::get<Ix>(std::forward<Arg>(arg))}...};
    }


    template<typename Arg>
    static constexpr auto
    tuple_to_range(Arg&& arg)
    {
      if constexpr (coordinates::pattern_range<Arg>)
        return std::forward<Arg>(arg);
      else
        return tuple_to_range_impl(std::forward<Arg>(arg), std::make_index_sequence<std::tuple_size_v<std::decay_t<Arg>>>{});
    }

  public:

    template<typename Arg>
    static constexpr auto
    component_collection(Arg&& arg)
    {
      if constexpr (coordinates::fixed_pattern<C> and values::fixed<Offset> and values::fixed<Extent>)
      {
        if constexpr (values::fixed_number_of_v<Extent> == 0)
        {
          return std::tuple {};
        }
        else if constexpr (coordinates::euclidean_pattern<C>)
        {
          return std::array {coordinates::Dimensions {arg.my_extent}};
        }
        else
        {
          return fixed_slice(coordinates::internal::get_component_collection(std::forward<Arg>(arg).my_coordinates));
        }
      }
      else
      {
        return coordinates::internal::SliceRange {
          tuple_to_range(coordinates::internal::get_component_collection(std::forward<Arg>(arg).my_coordinates)), arg.my_index};
      }
    }


#ifdef __cpp_concepts
    template<values::index N>
#else
    template<typename N, std::enable_if_t<values::index<N>, int> = 0>
#endif
    static constexpr auto
    component_start_indices(const T&, N n)
    {
      if constexpr (values::fixed<N>)
      {
        return component_start_indices_fixed(std::make_index_sequence<values::fixed_number_of_v<N>>{});
      }
      else
      {
        return component_start_indices_dynamic(n);
      }
    }


#ifdef __cpp_concepts
    template<values::index I>
#else
    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    static constexpr auto
#endif
    static constexpr auto
    index_table(const T& t, I i)
    {
      return coordinates::internal::get_index_table(t.my_coordinates, size(t) - i - 1_uz);
    }


#ifdef __cpp_concepts
    template<values::index I>
#else
    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    static constexpr auto
#endif
    static constexpr auto
    euclidean_index_table(const T& t, I i)
    {
      return coordinates::internal::get_euclidean_index_table(t.my_coordinates, stat_dimension(t) - i - 1_uz);
    }


    static constexpr auto
    stat_dimension(const T& t)
    {
      return values::operation {std::multiplies{}, coordinates::get_stat_dimension(t.my_coordinates), t.my_index};
    }


    static constexpr auto
    is_euclidean(const T& t)
    {
      return coordinates::get_is_euclidean(t.my_coordinates);
    }


#ifdef __cpp_concepts
    static constexpr values::value auto
    to_euclidean_component(const T& t, const auto& g, const values::index auto& euclidean_local_index)
    requires requires(std::size_t i){ {g(i)} -> values::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<values::index<L> and
      values::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
    {
      std::size_t n = coordinates::get_stat_dimension(t.my_coordinates);
      auto new_g = [&g, offset = euclidean_local_index / n](std::size_t i) { return g(offset + i); };
      return coordinates::to_stat_space(t.my_coordinates, new_g, euclidean_local_index % n);
    }


#ifdef __cpp_concepts
    static constexpr values::value auto
    from_euclidean_component(const T& t, const auto& g, const values::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> values::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<values::index<L> and
      values::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      auto n = coordinates::get_dimension(t.my_coordinates);
      auto new_g = [&g, offset = local_index / n](std::size_t i) { return g(offset + i); };
      return coordinates::from_stat_space(t.my_coordinates, new_g, local_index % n);
    }


#ifdef __cpp_concepts
    static constexpr values::value auto
    get_wrapped_component(const T& t, const auto& g, const values::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> values::value; }
#else
    template<typename Getter, typename L, std::enable_if_t<values::index<L> and
      values::value<typename std::invoke_result<const Getter&, std::size_t>::type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      auto n = coordinates::get_dimension(t.my_coordinates);
      auto new_g = [&g, offset = local_index / n](std::size_t i) { return g(offset + i); };
      return coordinates::get_wrapped_component(t.my_coordinates, new_g, local_index % n);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const values::value auto& x, const values::index auto& local_index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename X, typename L, std::enable_if_t<values::value<X> and values::index<L> and
      std::is_invocable<const Setter&, const X&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const X& x, const L& local_index)
#endif
    {
      using Scalar = std::decay_t<decltype(x)>;
      auto n = coordinates::get_dimension(t.my_coordinates);
      auto new_g = [&g, offset = local_index / n](std::size_t i) { return g(offset + i); };
      auto new_s = [&s, offset = local_index / n](const Scalar& x, std::size_t i) { s(x, offset + i); };
      return coordinates::set_wrapped_component(t.my_coordinates, new_s, new_g, x, local_index % n);
    }

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_DESCRIPTOR_SLICE_HPP
