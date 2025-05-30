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
 * \file
 * \internal
 * \brief Definition of \ref ranges::concat_view and \ref ranges::views::concat.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_CONCAT_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_CONCAT_HPP

#if __cpp_lib_ranges_concat < 202403L

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "view-concepts.hpp"
#include "view_interface.hpp"
#include "all.hpp"
#endif

namespace OpenKalman::ranges
{
  /**
   * \internal
   * \brief Equivalent to std::ranges::concat_view so long as all Views are std::ranges::random_access_range.
   * \sa views::concat
   */
#ifdef __cpp_lib_ranges
  template<std::ranges::random_access_range...Views> requires (... and std::ranges::view<Views>) and (sizeof...(Views) > 0)
  struct concat_view : std::ranges::view_interface<concat_view<Views...>>
#else
  template<typename...Views>
  struct concat_view : ranges::view_interface<concat_view<Views...>>
#endif
  {
  private:

#if __cplusplus >= 202002L
    template<typename...Rs> using concat_reference_t = std::common_reference_t<std::ranges::range_reference_t<Rs>...>;
    template<typename...Rs> using concat_value_t = std::common_type_t<std::ranges::range_value_t<Rs>...>;
    template<typename...Rs> using concat_rvalue_reference_t = std::common_reference_t<std::ranges::range_rvalue_reference_t<Rs>...>;
#else
    template<typename...Rs> using concat_reference_t = common_reference_t<ranges::range_reference_t<Rs>...>;
    template<typename...Rs> using concat_value_t = std::common_type_t<ranges::range_value_t<Rs>...>;
    template<typename...Rs> using concat_rvalue_reference_t = common_reference_t<ranges::range_rvalue_reference_t<Rs>...>;
#endif

    template<bool Const, typename Rs>
    using maybe_const = std::conditional_t<Const, const Rs, Rs>;

    using ViewsTup = std::tuple<Views...>;


    class concat_index_table
    {
      template<std::size_t...cs>
  #ifdef __cpp_lib_constexpr_vector
      static constexpr auto
  #else
      static auto
  #endif
      make_index_table_range(const ViewsTup& tup, std::index_sequence<cs...>)
      {
        std::vector<std::size_t> table;
        auto it = std::back_inserter(table);
#ifdef __cpp_lib_ranges
        (std::fill_n(it, std::ranges::size(std::get<cs>(tup)), cs), ...);
#else
        (std::fill_n(it, ranges::size(std::get<cs>(tup)), cs), ...);
#endif
        return table;
      }


      static constexpr auto
      make_index_table(const ViewsTup& tup)
      {
        return make_index_table_range(tup, std::index_sequence_for<Views...>{});
      }

    public:

      explicit constexpr concat_index_table(const ViewsTup& rs_tup) : value {make_index_table(rs_tup)} {}

      decltype(make_index_table(std::declval<const ViewsTup&>())) value;

    };


    class concat_start_table
    {
      template<std::size_t i = 0, typename...Locs>
      static constexpr auto
      make_start_table(const ViewsTup& tup, std::size_t currloc = 0_uz, Locs...locs)
      {
        if constexpr (i < sizeof...(Views))
        {
#ifdef __cpp_lib_ranges
          std::size_t next_loc = currloc + std::ranges::size(std::get<i>(tup));
#else
          std::size_t next_loc = currloc + ranges::size(std::get<i>(tup));
#endif
          return make_start_table<i + 1>(tup, next_loc, locs..., currloc);
        }
        else
        {
          return std::array {locs...};
        }
      }

    public:

      explicit constexpr concat_start_table(const ViewsTup& rs_tup) : value {make_start_table(rs_tup)} {}

      decltype(make_start_table(std::declval<const ViewsTup&>())) value;

    };


    using IndexTable = decltype(concat_index_table(std::declval<ViewsTup>()).value);

    using StartTable = decltype(concat_start_table(std::declval<ViewsTup>()).value);

  public:

    template<bool Const>
    struct iterator
    {
      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = iterator_concept;
      using value_type = concat_value_t<maybe_const<Const, Views>...>;
#if __cplusplus >= 202002L
      using difference_type = std::common_type_t<std::ranges::range_difference_t<maybe_const<Const, Views>>...>;
#else
      using difference_type = std::common_type_t<ranges::range_difference_t<maybe_const<Const, Views>>...>;
#endif
      using pointer = void;
      using reference = concat_reference_t<maybe_const<Const, Views>...>;

    private:

      using Tup = maybe_const<Const, ViewsTup>;

      template<typename = std::index_sequence_for<Views...>>
      struct tuple_concat_iterator_call_table;

      template<std::size_t...ix>
      struct tuple_concat_iterator_call_table<std::index_sequence<ix...>>
      {
        template<std::size_t i>
        static constexpr value_type
        call_table_get(Tup& tup, std::size_t local_i) noexcept
        {
          return std::get<i>(tup)[local_i];
        }

        static constexpr std::array value {call_table_get<ix>...};
      };

      using table = tuple_concat_iterator_call_table<>;

    public:

#ifdef __cpp_lib_ranges
      constexpr iterator() = default;

      constexpr iterator(iterator<not Const> it) requires Const and
        (... and std::convertible_to<std::ranges::iterator_t<Views>, std::ranges::iterator_t<const Views>>)
        : parent(it.parent), current(it.current) {}

      //template<typename...Args>
      constexpr explicit iterator(maybe_const<Const, concat_view>* parent, difference_type p)
        //requires std::constructible_from<base_iter, Args&&...>
        : parent(parent), current(p) {}
#else
      constexpr iterator() : parent{nullptr} {};

      template<bool C = Const, std::enable_if_t<C and
        (... and std::is_convertible_v<ranges::iterator_t<Views>, ranges::iterator_t<const Views>>), int> = 0>
      constexpr iterator(iterator<not C> it) : parent(it.parent), current(it.current) {}

      //template<typename...Args, bool Enable = true, std::enable_if_t<Enable and std::is_constructible_v<base_iter, Args&&...>, int> = 0>
      constexpr explicit iterator(maybe_const<Const, concat_view>* parent, difference_type p) : parent(parent), current(p) {}
#endif

      constexpr iterator(const iterator& other) = default;
      constexpr iterator(iterator&& other) noexcept = default;
      constexpr iterator& operator=(const iterator& other) = default;
      constexpr iterator& operator=(iterator&& other) noexcept = default;
      explicit constexpr operator std::size_t() const noexcept { return static_cast<std::size_t>(current); }

      constexpr decltype(auto) operator*() noexcept
      {
        std::size_t c = (parent->index_table)[static_cast<std::size_t>(current)];
        std::size_t local_i = current - parent->start_table[c];
        return table::value[c](parent->views_tup, local_i);
      }
      constexpr decltype(auto) operator*() const noexcept
      {
        std::size_t c = (parent->index_table)[static_cast<std::size_t>(current)];
        std::size_t local_i = current - parent->start_table[c];
        return table::value[c](parent->views_tup, local_i);
      }
      constexpr decltype(auto) operator[](difference_type offset) noexcept
      {
        auto i = static_cast<std::size_t>(current + offset);
        std::size_t c = parent->index_table[i];
        std::size_t local_i = i - parent->start_table[c];
        return table::value[c](parent->views_tup, local_i);
      }
      constexpr decltype(auto) operator[](difference_type offset) const noexcept
      {
        auto i = static_cast<std::size_t>(current + offset);
        std::size_t c = parent->index_table[i];
        std::size_t local_i = i - parent->start_table[c];
        return table::value[c](parent->views_tup, local_i);
      }
      constexpr auto& operator++() noexcept { ++current; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current -= diff; return *this; }

      friend constexpr auto operator+(const iterator& it, const difference_type diff) noexcept
      {
        return iterator {it.parent, it.current + diff};
      }
      friend constexpr auto operator+(const difference_type diff, const iterator& it) noexcept
      {
        return iterator {it.parent, diff + it.current};
      }
      friend constexpr auto operator-(const iterator& it, const difference_type diff) noexcept
      {
        return iterator {it.parent, it.current - diff};
      }
      friend constexpr difference_type operator-(const iterator& it, const iterator& other) noexcept
      {
        return it.current - other.current;
      }
      friend constexpr bool operator==(const iterator& it, const iterator& other) noexcept
      {
        return it.current == other.current;
      }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const iterator& other) const noexcept { return current <=> other.current; }
#else
      constexpr bool operator!=(const iterator& other) const noexcept { return current != other.current; }
      constexpr bool operator<(const iterator& other) const noexcept { return current < other.current; }
      constexpr bool operator>(const iterator& other) const noexcept { return current > other.current; }
      constexpr bool operator<=(const iterator& other) const noexcept { return current <= other.current; }
      constexpr bool operator>=(const iterator& other) const noexcept { return current >= other.current; }
#endif

    private:

      maybe_const<Const, concat_view>* parent;
      difference_type current;

    };


#ifdef __cpp_concepts
    constexpr concat_view() = default;
#else
    template<typename aT = void, std::enable_if_t<std::is_void_v<aT> and (... and std::is_default_constructible_v<Views>), int> = 0>
    constexpr concat_view() {}
#endif


    explicit constexpr concat_view(Views...vs) : views_tup {std::move(vs)...} {}


    constexpr auto
    begin() noexcept
    {
      return iterator<false> {this, 0};
    }


    constexpr auto
    begin() const noexcept
    {
      return iterator<true> {this, 0};
    }


    constexpr auto
    end() noexcept
    {
      return iterator<false> {this, static_cast<std::ptrdiff_t>(size())};
    }


    constexpr auto
    end() const noexcept
    {
      return iterator<true> {this, static_cast<std::ptrdiff_t>(size())};
    }

  private:

    template<typename F, typename Tuple>
    static constexpr auto tuple_transform(F&& f, Tuple&& tuple)
    {
      return std::apply([&f](auto&&...args)
      {
        return std::tuple<std::invoke_result_t<F&, decltype(args)>...>(std::invoke(f, std::forward<decltype(args)>(args))...);
      }, std::forward<Tuple>(tuple));

    }

  public:

#ifdef __cpp_lib_ranges
    constexpr auto size() const requires (... and std::ranges::sized_range<Views>)
#else
    template<typename aT = void, std::enable_if_t<std::is_void_v<aT> and (... and ranges::sized_range<Views>), int> = 0>
    constexpr auto size() const
#endif
    {
      return std::apply([](auto...sizes) { return (... + static_cast<std::size_t>(sizes)); },
#ifdef __cpp_lib_ranges
        tuple_transform(std::ranges::size, views_tup));
#else
        tuple_transform(ranges::size, views_tup));
#endif
    }

  private:

    ViewsTup views_tup;

    IndexTable index_table = concat_index_table(views_tup).value;

    StartTable start_table = concat_start_table(views_tup).value;

  };


  template<typename...Rs>
#ifdef __cpp_lib_ranges
  concat_view(Rs&&...) -> concat_view<std::ranges::views::all_t<Rs>...>;
#else
  concat_view(Rs&&...) -> concat_view<views::all_t<Rs>...>;
#endif

}


namespace OpenKalman::ranges::views
{
  namespace detail
  {
    struct concat_adaptor
    {
#ifdef __cpp_lib_ranges
      template<std::ranges::viewable_range...Rs> requires (... and std::ranges::random_access_range<Rs>)
#else
      template<typename...Rs, std::enable_if_t<(... and (ranges::viewable_range<Rs> and ranges::random_access_range<Rs>)), int> = 0>
#endif
      constexpr auto
      operator() (Rs&&...rs) const
      {
#ifdef __cpp_lib_ranges
        namespace ns = std;
#else
        namespace ns = OpenKalman;
#endif
        if constexpr (sizeof...(Rs) == 1 and (... and ns::ranges::input_range<Rs>))
          return ns::ranges::views::all(std::forward<Rs>(rs)...);
        else
          return concat_view {ns::ranges::views::all(std::forward<Rs>(rs))...};
      }
    };

  }


  /**
   * \brief Equivalent to std::ranges::views::concat so long as all Views are std::ranges::random_access_range.
   * \sa concat_view
   */
  inline constexpr detail::concat_adaptor concat;

}

#endif

#endif //OPENKALMAN_COMPATIBILITY_VIEWS_CONCAT_HPP
