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
 * \brief Definition of \ref collections::concat_view and \ref collections::views::concat.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_CONCAT_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_CONCAT_HPP

#include <tuple>
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/functions/get_collection_size.hpp"
#include "collections/functions/get.hpp"
#include "collection_view_interface.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename...Rs>
    class concat_index_table
    {
      template<std::size_t c = 0, std::size_t local = 0, std::size_t...cs>
      static constexpr auto
      make_index_table_tup(const std::tuple<Rs...>& tup, std::index_sequence<cs...> seq = std::index_sequence<>{})
      {
        if constexpr (c < sizeof...(Rs))
        {
          if constexpr (local < std::tuple_size_v<std::decay_t<std::tuple_element_t<c, std::tuple<Rs...>>>>)
            return make_index_table_tup<c, local + 1>(tup, std::index_sequence<cs..., c>{});
          else
            return make_index_table_tup<c + 1, 0>(tup, seq);
        }
        else
        {
          return std::tuple {std::integral_constant<std::size_t, cs>{}...};
        }
      }


      template<std::size_t...cs>
  #ifdef __cpp_lib_constexpr_vector
      static constexpr auto
  #else
      static auto
  #endif
      make_index_table_range(const std::tuple<Rs...>& tup, std::index_sequence<cs...>)
      {
        std::vector<std::size_t> table;
        auto it = std::back_inserter(table);
        (std::fill_n(it, get_collection_size(std::get<cs>(tup)), cs), ...);
        return table;
      }


      static constexpr auto
      make_index_table(const std::tuple<Rs...>& tup)
      {
        static_assert((... and collection<Rs>));
        if constexpr ((... and tuple_like<Rs>)) return make_index_table_tup(tup);
        else return make_index_table_range(tup, std::index_sequence_for<Rs...>{});
      }

    public:

      explicit constexpr concat_index_table(const std::tuple<Rs...>& rs_tup) : value {make_index_table(rs_tup)} {}

      decltype(make_index_table(std::declval<const std::tuple<Rs...>&>())) value;

    };


    template<typename...Rs>
    concat_index_table(const std::tuple<Rs...>&) -> concat_index_table<Rs...>;


    template<typename...Rs>
    class concat_start_table
    {
      template<std::size_t i = 0, typename CurrLoc = std::integral_constant<std::size_t, 0>, typename...Locs>
      static constexpr auto make_start_table(const std::tuple<Rs...>& tup, CurrLoc currloc = {}, Locs...locs)
      {
        if constexpr (i < sizeof...(Rs))
        {
          auto next_loc = value::operation {std::plus{}, currloc, get_collection_size(std::get<i>(tup))};
          return make_start_table<i + 1>(tup, std::move(next_loc), std::move(locs)..., std::move(currloc));
        }
        else if constexpr ((... and tuple_like<Rs>))
        {
          return std::tuple {std::move(locs)...};
        }
        else
        {
          return std::array {static_cast<std::size_t>(std::move(locs))...};
        }
      }

    public:

      explicit constexpr concat_start_table(const std::tuple<Rs...>& rs_tup) : value {make_start_table(rs_tup)} {}

      decltype(make_start_table(std::declval<const std::tuple<Rs...>&>())) value;

    };


    template<typename...Rs>
    concat_start_table(const std::tuple<Rs...>&) -> concat_start_table<Rs...>;

  } // namespace detail


  namespace internal
  {
    namespace detail
    {
      template<typename Tup, typename = std::make_index_sequence<std::tuple_size_v<Tup>>>
      struct tuple_concat_iterator_call_table;

      template<typename Tup, std::size_t...ix>
      struct tuple_concat_iterator_call_table<Tup, std::index_sequence<ix...>>
      {
#if __cplusplus >= 202002L
        using element_type = std::common_reference_t<const common_collection_type_t<std::tuple_element_t<ix, Tup>>&...>;
#else
        using element_type = std::common_type_t<const common_collection_type_t<std::tuple_element_t<ix, Tup>>&...>;
#endif

        template<std::size_t i>
        static constexpr element_type
        call_table_get(const Tup& tup, std::size_t local_i) noexcept
        {
          return get(OpenKalman::internal::generalized_std_get<i>(tup), local_i);
        }

        static constexpr std::array value {call_table_get<ix>...};
      };
    }


    /**
     * \internal
     * \brief Iterator for \ref concat_view
     */
    template<typename RsTup, typename IndexTable, typename StartTable>
    struct concat_view_iterator
    {
    private:

      using table = detail::tuple_concat_iterator_call_table<RsTup>;

    public:

      using iterator_category = std::random_access_iterator_tag;
      using value_type = typename table::element_type;
      using difference_type = std::ptrdiff_t;
      explicit constexpr concat_view_iterator(const RsTup& rs_tup, const IndexTable& it, const StartTable& st, std::size_t p) noexcept
        : rs_tup_ptr {std::addressof(rs_tup)}, index_table_ptr{std::addressof(it)}, start_table_ptr{std::addressof(st)},
          current{static_cast<difference_type>(p)} {}
      constexpr concat_view_iterator() = default;
      constexpr concat_view_iterator(const concat_view_iterator& other) = default;
      constexpr concat_view_iterator(concat_view_iterator&& other) noexcept = default;
      constexpr concat_view_iterator& operator=(const concat_view_iterator& other) = default;
      constexpr concat_view_iterator& operator=(concat_view_iterator&& other) noexcept = default;
      explicit constexpr operator std::size_t() const noexcept { return static_cast<std::size_t>(current); }

      constexpr decltype(auto) operator*() noexcept
      {
        std::size_t c = get(*index_table_ptr, static_cast<std::size_t>(current));
        std::size_t local_i = current - get(*start_table_ptr, c);
        return table::value[c](*rs_tup_ptr, local_i);
      }
      constexpr decltype(auto) operator*() const noexcept
      {
        std::size_t c = get(*index_table_ptr, static_cast<std::size_t>(current));
        std::size_t local_i = current - get(*start_table_ptr, c);
        return table::value[c](*rs_tup_ptr, local_i);
      }
      constexpr decltype(auto) operator[](difference_type offset)
      {
        if (current + offset < 0) throw std::out_of_range {"Offset to iterator out of range"};
        auto i = static_cast<std::size_t>(current + offset);
        std::size_t c = get(*index_table_ptr, i);
        std::size_t local_i = i - get(*start_table_ptr, c);
        return table::value[c](*rs_tup_ptr, local_i);
      }
      constexpr decltype(auto) operator[](difference_type offset) const
      {
        if (current + offset < 0) throw std::out_of_range {"Offset to iterator out of range"};
        auto i = static_cast<std::size_t>(current + offset);
        std::size_t c = get(*index_table_ptr, i);
        std::size_t local_i = i - get(*start_table_ptr, c);
        return table::value[c](*rs_tup_ptr, local_i);
      }
      constexpr auto& operator++() noexcept { ++current; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current -= diff; return *this; }

      friend constexpr auto operator+(const concat_view_iterator& it, const difference_type diff) noexcept
      {
        return concat_view_iterator {*it.rs_tup_ptr, *it.index_table_ptr, *it.start_table_ptr, static_cast<std::size_t>(it.current + diff)};
      }
      friend constexpr auto operator+(const difference_type diff, const concat_view_iterator& it) noexcept
      {
        return concat_view_iterator {*it.rs_tup_ptr, *it.index_table_ptr, *it.start_table_ptr, static_cast<std::size_t>(diff + it.current)};
      }
      friend constexpr auto operator-(const concat_view_iterator& it, const difference_type diff)
      {
        if (it.current < diff) throw std::out_of_range {"Iterator out of range"};
        return concat_view_iterator {*it.rs_tup_ptr, *it.index_table_ptr, *it.start_table_ptr, static_cast<std::size_t>(it.current - diff)};
      }
      friend constexpr difference_type operator-(const concat_view_iterator& it, const concat_view_iterator& other) noexcept
      {
        return it.current - other.current;
      }
      friend constexpr bool operator==(const concat_view_iterator& it, const concat_view_iterator& other) noexcept
      {
        return it.current == other.current;
      }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const concat_view_iterator& other) const noexcept { return current <=> other.current; }
#else
      constexpr bool operator!=(const concat_view_iterator& other) const noexcept { return current != other.current; }
      constexpr bool operator<(const concat_view_iterator& other) const noexcept { return current < other.current; }
      constexpr bool operator>(const concat_view_iterator& other) const noexcept { return current > other.current; }
      constexpr bool operator<=(const concat_view_iterator& other) const noexcept { return current <= other.current; }
      constexpr bool operator>=(const concat_view_iterator& other) const noexcept { return current >= other.current; }
#endif

    private:

      const RsTup* rs_tup_ptr; // \todo Convert this to std::shared_ptr<Derived> if p3037Rx ("constexpr std::shared_ptr") is adopted

      const IndexTable* index_table_ptr;

      const StartTable* start_table_ptr;

      difference_type current;

    };


    template<typename RsTup, typename IndexTable, typename StartTable>
    concat_view_iterator(const RsTup&, const IndexTable&, const StartTable&, std::size_t)
      -> concat_view_iterator<RsTup, IndexTable, StartTable>;

  }


  /**
   * \internal
   * \brief A view that concatenates some number of \ref collection "collections".
   * \details
   * The following should compile:
   * \code
   * static_assert(std::tuple_size_v<concat_view<std::tuple<>, std::tuple<>>> == 0);
   * static_assert(std::tuple_size_v<concat_view<std::tuple<double, int>, std::tuple<unsigned, std::tuple<>, float>>> == 5);
   * static_assert(std::is_same_v<std::tuple_element_t<1, concat_view<std::tuple<double, int>, std::tuple<unsigned, std::tuple<>, float>>>, int>);
   * static_assert(std::is_same_v<std::tuple_element_t<3, concat_view<std::tuple<double, int>, std::tuple<unsigned, std::tuple<>, float>>>, std::tuple<>>);
   * static_assert(collections::get<2>(concat_view {std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}}) == 6.f);
   * static_assert(collections::get<3>(concat_view {std::tuple{4, 5.}, std::tuple{6.f, std::tuple{}}}) == std::tuple{});
   * static_assert((concat_view {std::vector{3, 4, 5}, std::vector{6, 7, 8}}[2u]), 5);
   * static_assert((concat_view {std::vector{3, 4, 5}, std::vector{6, 7, 8}}[std::integral_constant<std::size_t, 3>{}]), 6);
   * \endcode
   * \sa views::concat
   */
#ifdef __cpp_lib_ranges
  template<collection...Ts> requires (sizeof...(Ts) > 0)
#else
  template<typename...Ts>
#endif
  struct concat_view : collection_view_interface<concat_view<Ts...>>
  {
  private:

    static constexpr bool all_tuples = (... and tuple_like<Ts>);

    using MyTsTup = std::tuple<all_view<Ts>...>;


    template<std::size_t i>
    constexpr decltype(auto) get_t() & noexcept
    {
      using Ti = std::tuple_element_t<i, MyTsTup>;
      return std::forward<Ti&>(std::get<i>(my_ts));
    }

    template<std::size_t i>
    constexpr decltype(auto) get_t() const & noexcept
    {
      using Ti = std::tuple_element_t<i, MyTsTup>;
      return std::forward<const Ti&>(std::get<i>(my_ts));
    }

    template<std::size_t i>
    constexpr decltype(auto) get_t() && noexcept
    {
      using Ti = std::tuple_element_t<i, MyTsTup>;
      return std::forward<Ti&&>(std::get<i>(my_ts));
    }

    template<std::size_t i>
    constexpr decltype(auto) get_t() const && noexcept
    {
      using Ti = std::tuple_element_t<i, MyTsTup>;
      return std::forward<const Ti&&>(std::get<i>(my_ts));
    }

  public:

#ifdef __cpp_concepts
    constexpr concat_view() requires std::default_initializable<MyTsTup> = default;
#else
    template<typename aT = void, std::enable_if_t<std::is_void_v<aT> and std::is_default_constructible_v<MyTsTup>, int> = 0>
    constexpr concat_view() {}
#endif


#ifdef __cpp_concepts
    template<typename...Args> requires std::constructible_from<MyTsTup, Args&&...>
#else
    template<typename...Args, std::enable_if_t<std::is_constructible_v<MyTsTup, Args&&...>, int> = 0>
#endif
    explicit constexpr concat_view(Args&&...args) : my_ts {std::forward<Args>(args)...} {}


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i> requires all_tuples
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>), "Index out of range");
      auto c = collections::get(std::forward<decltype(self)>(self).index_table, std::integral_constant<std::size_t, i>{});
      auto local_i = i - collections::get(std::forward<decltype(self)>(self).start_table, c);
      return collections::get(std::forward<decltype(self)>(self).template get_t<c>(), local_i);
    }
#else
    template<std::size_t i, bool atp = all_tuples, std::enable_if_t<atp, int> = 0>
    constexpr decltype(auto)
    get() &
    {
      static_assert(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>), "Index out of range");
      auto c = collections::get(index_table, std::integral_constant<std::size_t, i>{});
      auto local_i = i - collections::get(start_table, c);
      return collections::get(get_t<c>(), local_i);
    }

    template<std::size_t i, bool atp = all_tuples, std::enable_if_t<atp, int> = 0>
    constexpr decltype(auto)
    get() const &
    {
      static_assert(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>), "Index out of range");
      auto c = collections::get(index_table, std::integral_constant<std::size_t, i>{});
      auto local_i = i - collections::get(start_table, c);
      return collections::get(get_t<c>(), local_i);
    }

    template<std::size_t i, bool atp = all_tuples, std::enable_if_t<atp, int> = 0>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>), "Index out of range");
      auto c = collections::get(index_table, std::integral_constant<std::size_t, i>{});
      auto local_i = i - collections::get(std::move(*this).start_table, c);
      return collections::get(std::move(*this).template get_t<c>(), local_i);

    }

    template<std::size_t i, bool atp = all_tuples, std::enable_if_t<atp, int> = 0>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert(i < (0_uz + ... + std::tuple_size_v<std::decay_t<Ts>>), "Index out of range");
      auto c = collections::get(index_table, std::integral_constant<std::size_t, i>{});
      auto local_i = i - collections::get(std::move(*this).start_table, c);
      return collections::get(std::move(*this).template get_t<c>(), local_i);
    }
#endif

  private:

    struct size_impl
    {
      constexpr auto operator()() const { return std::integral_constant<std::size_t, 0> {}; }

      template<typename U, typename...Us>
      constexpr auto operator()(const U& u, const Us&... us) const
      {
        return value::operation {std::plus{}, get_collection_size(u), operator()(us...)};
      }
    };

  public:

#ifdef __cpp_concepts
    constexpr value::index auto size() const
#else
    constexpr auto size() const
#endif
    {
      return std::apply(size_impl{}, my_ts);
    }


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
#ifdef __cpp_concepts
    constexpr auto
    begin() noexcept requires (not all_tuples)
#else
    template<bool alltup = all_tuples, std::enable_if_t<(not alltup), int> = 0>
    constexpr auto
    begin() noexcept
#endif
    {
      return internal::concat_view_iterator {my_ts, index_table, start_table, 0};
    }


    /// \overload
#ifdef __cpp_concepts
    constexpr auto
    begin() const noexcept requires (not all_tuples)
#else
    template<bool alltup = all_tuples, std::enable_if_t<(not alltup), int> = 0>
    constexpr auto
    begin() const noexcept
#endif
    {
      return internal::concat_view_iterator {my_ts, index_table, start_table, 0};
    }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_concepts
    constexpr auto
    end() noexcept requires (not all_tuples)
#else
    template<bool alltup = all_tuples, std::enable_if_t<(not alltup), int> = 0>
    constexpr auto
    end() noexcept
#endif
    {
      return internal::concat_view_iterator {my_ts, index_table, start_table, value::to_number(size())};
    }


    /// \overload
#ifdef __cpp_concepts
    constexpr auto
    end() const noexcept requires (not all_tuples)
#else
    template<bool alltup = all_tuples, std::enable_if_t<(not alltup), int> = 0>
    constexpr auto
    end() const noexcept
#endif
    {
      return internal::concat_view_iterator {my_ts, index_table, start_table, value::to_number(size())};
    }

  private:

    MyTsTup my_ts;

    decltype(detail::concat_index_table(my_ts).value) index_table = detail::concat_index_table(my_ts).value;

    decltype(detail::concat_start_table(my_ts).value) start_table = detail::concat_start_table(my_ts).value;

  };


  /**
   * \brief Deduction guide
   */
  template<typename...Args>
  concat_view(Args&&...) -> concat_view<Args...>;


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct concat_tuple_size {};

#ifdef __cpp_concepts
    template<typename...Ts> requires (... and (size_of_v<Ts> != dynamic_size))
    struct concat_tuple_size<concat_view<Ts...>>
#else
    template<typename...Ts>
    struct concat_tuple_size<concat_view<Ts...>, std::enable_if_t<(... and (size_of<Ts>::value != dynamic_size))>>
#endif
      : std::integral_constant<std::size_t, (0_uz + ... + size_of_v<Ts>)> {};


#ifdef __cpp_concepts
    template<std::size_t i, typename T>
#else
    template<std::size_t i, typename T, typename = void>
#endif
    struct concat_tuple_element {};

#ifdef __cpp_concepts
    template<std::size_t i, typename...Ts> requires
      requires(concat_view<Ts...>& v) { OpenKalman::internal::generalized_std_get<i>(v); }
    struct concat_tuple_element<i, concat_view<Ts...>>
#else
    template<std::size_t i, typename...Ts>
    struct concat_tuple_element<i, concat_view<Ts...>,
      std::void_t<decltype(OpenKalman::internal::generalized_std_get<i>(std::declval<concat_view<Ts...>&>()))>>
#endif
    {
    private:
      using V = concat_view<Ts...>;
      using MyTsTup = std::tuple<all_view<Ts>...>;
      static constexpr std::size_t c = value::fixed_number_of_v<decltype(std::get<i>(concat_index_table(std::declval<MyTsTup>()).value))>;
      static constexpr std::size_t local_i = i - value::fixed_number_of_v<decltype(std::get<c>(concat_start_table(std::declval<MyTsTup>()).value))>;
    public:
      using type = std::tuple_element_t<local_i, std::decay_t<std::tuple_element_t<c, MyTsTup>>>;
    };
  }

} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename...Rs>
  constexpr bool enable_borrowed_range<OpenKalman::collections::concat_view<Rs...>> =
    (... and (std::is_lvalue_reference_v<Rs> or enable_borrowed_range<remove_cvref_t<Rs>>));
}


namespace std
{
  template<typename...Ts>
  struct tuple_size<OpenKalman::collections::concat_view<Ts...>>
    : OpenKalman::collections::detail::concat_tuple_size<OpenKalman::collections::concat_view<Ts...>> {};

  template<size_t i, typename...Ts>
  struct tuple_element<i, OpenKalman::collections::concat_view<Ts...>>
    : OpenKalman::collections::detail::concat_tuple_element<i, OpenKalman::collections::concat_view<Ts...>> {};

  template<typename RsTup, typename IndexTable, typename StartTable>
  struct iterator_traits<OpenKalman::collections::internal::concat_view_iterator<RsTup, IndexTable, StartTable>>
  {
    using difference_type = typename OpenKalman::collections::internal::concat_view_iterator<RsTup, IndexTable, StartTable>::difference_type;
    using value_type = typename OpenKalman::collections::internal::concat_view_iterator<RsTup, IndexTable, StartTable>::value_type;
    using iterator_category = typename OpenKalman::collections::internal::concat_view_iterator<RsTup, IndexTable, StartTable>::iterator_category;
  };

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct concat_impl
    {
#ifdef __cpp_concepts
      template<collection...R>
#else
      template<typename...R, std::enable_if_t<(... and collection<R>), int> = 0>
#endif
      constexpr auto
      operator() [[nodiscard]] (R&&...r) const { return concat_view<R...> {std::forward<R>(r)...}; }
    };







#ifdef __cpp_concepts
    template<viewable_collection...Rs>
#else
    template<typename...Rs>
#endif
    struct concat_closure
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<concat_closure<Rs...>>
#endif
    {
      constexpr concat_closure(Rs&&...rs) : rs_tup {std::forward<Rs>(rs)...} {};


#ifdef __cpp_explicit_this_parameter
      template<viewable_collection R>
      constexpr auto
      operator() (this auto&& self, R&& r) noexcept
      {
        return std::apply([](R&& r, Rs&&...rs){ return concat_view {std::forward<R>(r), std::forward<Rs>(rs)...}; },
          std::tuple_cat(std::forward_as_tuple(std::forward<R>(r)), std::forward<decltype(self)>(self).rs_tup));
      }
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
      constexpr auto
      operator() (R&& r) & noexcept
      {
        return std::apply([](R&& r, Rs&&...rs){ return concat_view {std::forward<R>(r), std::forward<Rs>(rs)...}; },
          std::tuple_cat(std::forward_as_tuple(std::forward<R>(r)), rs_tup));
      }


      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
      constexpr auto
      operator() (R&& r) const & noexcept
      {
        return std::apply([](R&& r, Rs&&...rs){ return concat_view {std::forward<R>(r), std::forward<Rs>(rs)...}; },
          std::tuple_cat(std::forward_as_tuple(std::forward<R>(r)), rs_tup));
      }


      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
      constexpr auto
      operator() (R&& r) && noexcept
      {
        return std::apply([](R&& r, Rs&&...rs){ return concat_view {std::forward<R>(r), std::forward<Rs>(rs)...}; },
          std::tuple_cat(std::forward_as_tuple(std::forward<R>(r)), std::move(rs_tup)));
      }


      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
      constexpr auto
      operator() (R&& r) const && noexcept
      {
        return std::apply([](R&& r, Rs&&...rs){ return concat_view {std::forward<R>(r), std::forward<Rs>(rs)...}; },
          std::tuple_cat(std::forward_as_tuple(std::forward<R>(r)), std::move(rs_tup)));
      }
#endif

    private:

      std::tuple<Rs...> rs_tup;

    };


    struct concat_adaptor
    {
#ifdef __cpp_concepts
      template<viewable_collection...Rs>
#else
      template<typename...Rs, std::enable_if_t<(...and viewable_collection<Rs>), int> = 0>
#endif
      constexpr auto
      operator() (Rs&&...rs) const
      {
        return concat_closure<Rs...>{std::forward<Rs>(rs)...};
      }


#ifdef __cpp_concepts
      template<viewable_collection R, viewable_collection...Rs>
#else
      template<typename R, typename...Rs, std::enable_if_t<(viewable_collection<R> and ... and viewable_collection<Rs>), int> = 0>
#endif
      constexpr auto
      operator() (R&& r, Rs&&...rs) const
      {
        return concat_view {std::forward<R>(r), std::forward<Rs>(rs)...};
      }
    };

  }


  /**
   * \brief a RangeAdapterObject associated with \ref concat_view.
   * \details The expression <code>views::concat(arg)</code> is expression-equivalent
   * to <code>concat_view(arg)</code> for any suitable \ref collection arg.
   * \sa concat_view
   */
  inline constexpr detail::concat_adaptor concat;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_CONCAT_HPP
