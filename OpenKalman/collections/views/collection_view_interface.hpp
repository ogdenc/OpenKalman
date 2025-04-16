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
 * \brief Definition of \ref collection_view_interface
 */

#ifndef OPENKALMAN_COLLECTION_VIEW_HPP
#define OPENKALMAN_COLLECTION_VIEW_HPP

#include <type_traits>
#include <tuple>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "basics/classes/movable_wrapper.hpp"
#include "values/concepts/fixed.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"

namespace OpenKalman::collections
{
  /**
   * \internal
   * \brief A CRTP helper class template for defining a \ref collection_view.
   * \details The derived class must define at least the following:
   * \code
   * constexpr decltype(auto) operator[](value::index auto i) const;
   * constexpr value::index auto size() const;
   * \endcode
   * If the <code>size</code> function returns a \rev value::fixed "fixed" index and the subscript <code>operator[]</code> function
   * returns a value when \ref value::index "index" i is \ref value::fixed "fixed", the resulting view will be \ref tuple_like.
   * Independently of this, if the subscript <code>operator[]</code> function returns a value when
   * \ref value::index "index" i is \ref value::dynamic "dynamic", the resulting view will be a \ref sized_random_access_range.
   * \tparam Derived The derived view class.
   * \sa collection_view
   */
  #ifdef __cpp_lib_ranges
  template<typename Derived> requires std::is_class_v<Derived> and std::same_as<Derived, std::remove_cv_t<Derived>>
  struct collection_view_interface;
  #else
  template<typename Derived>
  struct collection_view_interface;
  #endif


  namespace internal
  {
    namespace detail
    {
      template<typename D, typename = std::make_index_sequence<std::tuple_size_v<D>>>
      struct tuple_iterator_call_table;

      template<typename D, std::size_t...ix>
      struct tuple_iterator_call_table<D, std::index_sequence<ix...>>
      {
#if __cplusplus >= 202002L
        using element_type = std::common_reference_t<const std::tuple_element_t<ix, D>&...>;
#else
        using element_type = std::common_type_t<const std::tuple_element_t<ix, D>&...>;
#endif

        template<std::size_t i>
        static constexpr element_type
        call_table_get(const D& tup) noexcept { return OpenKalman::internal::generalized_std_get<i>(tup); }

        static constexpr std::array value {call_table_get<ix>...};
      };
    }


    /**
     * \internal
     * \brief Iterator for \ref collection_view_interface
     */
    template<typename D>
    struct tuple_iterator
    {
    private:

      using table = detail::tuple_iterator_call_table<std::decay_t<D>>;

    public:

      using iterator_category = std::random_access_iterator_tag;
      using value_type = std::remove_cv_t<typename table::element_type>;
      using difference_type = std::ptrdiff_t;
      explicit constexpr tuple_iterator(D& d, std::size_t p) noexcept : d_ptr {std::addressof(d)}, current{static_cast<difference_type>(p)} {}
      constexpr tuple_iterator() = default;
      constexpr tuple_iterator(const tuple_iterator& other) = default;
      constexpr tuple_iterator(tuple_iterator&& other) noexcept = default;
      constexpr tuple_iterator& operator=(const tuple_iterator& other) = default;
      constexpr tuple_iterator& operator=(tuple_iterator&& other) noexcept = default;
      explicit constexpr operator std::size_t() const noexcept { return static_cast<std::size_t>(current); }
      constexpr decltype(auto) operator*() noexcept { return table::value[current](*d_ptr); }
      constexpr decltype(auto) operator*() const noexcept { return table::value[current](*d_ptr); }
      constexpr decltype(auto) operator[](difference_type offset) noexcept { return table::value[current + offset](*d_ptr); }
      constexpr decltype(auto) operator[](difference_type offset) const noexcept { return table::value[current + offset](*d_ptr); }
      constexpr auto& operator++() noexcept { ++current; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current -= diff; return *this; }
      friend constexpr auto operator+(const tuple_iterator& it, const difference_type diff) noexcept { return tuple_iterator {*it.d_ptr, static_cast<std::size_t>(it.current + diff)}; }
      friend constexpr auto operator+(const difference_type diff, const tuple_iterator& it) noexcept { return tuple_iterator {*it.d_ptr, static_cast<std::size_t>(diff + it.current)}; }
      friend constexpr auto operator-(const tuple_iterator& it, const difference_type diff) { if (it.current < diff) throw std::out_of_range{"Iterator out of range"}; return tuple_iterator {*it.d_ptr, static_cast<std::size_t>(it.current - diff)}; }
      friend constexpr difference_type operator-(const tuple_iterator& it, const tuple_iterator& other) noexcept { return it.current - other.current; }
      friend constexpr bool operator==(const tuple_iterator& it, const tuple_iterator& other) noexcept { return it.current == other.current; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const tuple_iterator& other) const noexcept { return current <=> other.current; }
#else
      constexpr bool operator!=(const tuple_iterator& other) const noexcept { return current != other.current; }
      constexpr bool operator<(const tuple_iterator& other) const noexcept { return current < other.current; }
      constexpr bool operator>(const tuple_iterator& other) const noexcept { return current > other.current; }
      constexpr bool operator<=(const tuple_iterator& other) const noexcept { return current <= other.current; }
      constexpr bool operator>=(const tuple_iterator& other) const noexcept { return current >= other.current; }
#endif

    private:

      D * d_ptr; // \todo Convert this to std::shared_ptr<Derived> if p3037Rx ("constexpr std::shared_ptr") is adopted
      difference_type current;

    }; // struct Iterator


    template<typename D>
    tuple_iterator(D&&, std::size_t) -> tuple_iterator<std::remove_reference_t<D>>;

  }


#ifdef __cpp_lib_ranges
  template<typename Derived> requires std::is_class_v<Derived> and std::same_as<Derived, std::remove_cv_t<Derived>>
  struct collection_view_interface : std::ranges::view_interface<Derived>
#else
  template<typename Derived>
  struct collection_view_interface : ranges::view_interface<Derived>
#endif
  {
  private:

#ifdef __cpp_lib_ranges
    using base = std::ranges::view_interface<Derived>;
#else
    using base = ranges::view_interface<Derived>;
#endif

  public:
    /**
     * \brief Get element i the derived object, assuming it is a fixed-size range.
     * \details This effectively makes a range \ref tuple_like.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i, typename Self>
    constexpr decltype(auto)
    get(this Self&& self) noexcept
    {
      return collections::get(std::forward<Self>(self), std::integral_constant<std::size_t, i>{});
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      return collections::get(static_cast<Derived&>(*this), std::integral_constant<std::size_t, i>{});
    }


    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      return collections::get(static_cast<const Derived&>(*this), std::integral_constant<std::size_t, i>{});
    }


    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      return collections::get(static_cast<Derived&&>(*this), std::integral_constant<std::size_t, i>{});
    }


    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      return collections::get(static_cast<const Derived&&>(*this), std::integral_constant<std::size_t, i>{});
    }
#endif


    /**
     * \brief The size of the base object.
     */
#ifdef __cpp_concepts
    constexpr value::index auto
    size() const noexcept
#else
    constexpr auto size() const noexcept
#endif
    {
      if constexpr (internal::size_if_collection_v<Derived> == dynamic_size)
      {
#ifdef __cpp_lib_ranges
        return std::ranges::size(static_cast<const Derived&>(*this));
#else
        return ranges::size(static_cast<const Derived&>(*this));
#endif
      }
      else
      {
        return internal::size_if_collection<Derived> {};
      }
    }


    /**
     * \returns An iterator at the beginning
     * \note If the base object is \ref tuple_like and not a \ref sized_random_access_range, this will incur
     * overhead (once per execution) that is linear in the number of elements.
     */
    friend constexpr auto
    begin(collection_view_interface& d) noexcept
    {
      return internal::tuple_iterator {static_cast<Derived&>(d), 0_uz};
    }

    /// \overload
    friend constexpr auto
    begin(const collection_view_interface& d) noexcept
    {
      return internal::tuple_iterator {static_cast<const Derived&>(d), 0_uz};
    }


    /**
     * \returns An iterator at the end, if the base object is \ref tuple_like.
     * \note If the base object is \ref tuple_like and not a \ref sized_random_access_range, this will incur
     * overhead (once per execution) that is linear in the number of elements.
     */
    friend constexpr auto
    end(collection_view_interface& d) noexcept
    {
      return internal::tuple_iterator {static_cast<Derived&>(d), std::tuple_size_v<Derived>};
    }

    /// \overload
    friend constexpr auto
    end(const collection_view_interface& d) noexcept
    {
      return internal::tuple_iterator {static_cast<const Derived&>(d), std::tuple_size_v<Derived>};
    }


    /**
     * \returns Returns the initial element.
     */
    constexpr decltype(auto)
    front()
    {
      return collections::get(static_cast<Derived&>(*this), std::integral_constant<std::size_t, 0>{});
    }

    /// \overload
    constexpr decltype(auto)
    front() const
    {
      return collections::get(static_cast<const Derived&>(*this), std::integral_constant<std::size_t, 0>{});
    }


    /**
     * \returns Returns the final element.
     */
    constexpr decltype(auto)
    back()
    {
      return collections::get(static_cast<Derived&>(*this), value::operation {std::minus{}, size(), std::integral_constant<std::size_t, 1>{}});
    }

    /// \overload
    constexpr decltype(auto)
    back() const
    {
      return collections::get(static_cast<const Derived&>(*this), value::operation {std::minus{}, size(), std::integral_constant<std::size_t, 1>{}});
    }


    /**
     * \brief Subscript operator
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self, value::index I>
    constexpr decltype(auto)
    operator[](this Self&& self, I i) noexcept
    {
      if constexpr (internal::size_if_collection_v<Self> != dynamic_size and value::fixed<I>)
        static_assert(value::fixed_number_of<I>::value < internal::size_if_collection<Self>::value, "Index out of range");

      if constexpr (value::fixed<I> or sized_random_access_range<Self>)
        return collections::get(std::forward<Self>(self), std::move(i));
      else
        return internal::tuple_iterator {self, 0_uz}[value::to_number(std::move(i))];
    }
#else
    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &
    {
      if constexpr (internal::size_if_collection_v<Derived> != dynamic_size and value::fixed<I>)
        static_assert(value::fixed_number_of<I>::value < internal::size_if_collection<Derived>::value, "Index out of range");

      if constexpr (value::fixed<I> or sized_random_access_range<Derived>)
        return collections::get(static_cast<Derived&>(*this), std::move(i));
      else
        return internal::tuple_iterator {static_cast<Derived&>(*this), 0_uz}[value::to_number(std::move(i))];
    }

    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr (internal::size_if_collection_v<Derived> != dynamic_size and value::fixed<I>)
        static_assert(value::fixed_number_of<I>::value < internal::size_if_collection<Derived>::value, "Index out of range");

      if constexpr (value::fixed<I> or sized_random_access_range<Derived>)
        return collections::get(static_cast<const Derived&>(*this), std::move(i));
      else
        return internal::tuple_iterator {static_cast<const Derived&>(*this), 0_uz}[value::to_number(std::move(i))];
    }

    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) && noexcept
    {
      if constexpr (internal::size_if_collection_v<Derived> != dynamic_size and value::fixed<I>)
        static_assert(value::fixed_number_of<I>::value < internal::size_if_collection<Derived>::value, "Index out of range");

      if constexpr (value::fixed<I> or sized_random_access_range<Derived>)
        return collections::get(static_cast<Derived&&>(*this), std::move(i));
      else
        return internal::tuple_iterator {static_cast<Derived&&>(*this), 0_uz}[value::to_number(std::move(i))];
    }

    template<typename I, std::enable_if_t<value::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const && noexcept
    {
      if constexpr (internal::size_if_collection_v<Derived> != dynamic_size and value::fixed<I>)
        static_assert(value::fixed_number_of<I>::value < internal::size_if_collection<Derived>::value, "Index out of range");

      if constexpr (value::fixed<I> or sized_random_access_range<Derived>)
        return collections::get(static_cast<const Derived&&>(*this), std::move(i));
      else
        return internal::tuple_iterator {static_cast<const Derived&&>(*this), 0_uz}[value::to_number(std::move(i))];
    }
#endif

  };


#ifdef __cpp_impl_three_way_comparison
  namespace detail
  {
    template<std::size_t i = 0, typename T1, typename T2>
    constexpr std::partial_ordering
    fixed_compare(const T1& lhs, const T2& rhs)
    {
      if constexpr (i == size_of_v<T1> and i == size_of_v<T2>) return std::partial_ordering::equivalent;
      else if constexpr (i == size_of_v<T1>) return std::partial_ordering::less;
      else if constexpr (i == size_of_v<T2>) return std::partial_ordering::greater;
      else if (get(lhs, std::integral_constant<std::size_t, i>{}) == get(rhs, std::integral_constant<std::size_t, i>{}))
        return fixed_compare<i + 1>(lhs, rhs);
      else return get(lhs, std::integral_constant<std::size_t, i>{}) <=> get(rhs, std::integral_constant<std::size_t, i>{});
    }

    template<typename T1, typename T2>
    constexpr std::partial_ordering
    compare_impl(const T1& lhs, const T2& rhs)
    {
      if constexpr (size_of_v<T1> != dynamic_size and size_of_v<T2> != dynamic_size) return fixed_compare(lhs, rhs);
      else return std::lexicographical_compare_three_way(
        std::ranges::begin(lhs), std::ranges::end(lhs),
        std::ranges::begin(rhs), std::ranges::end(rhs));
    }
  }


  template<collection Lhs, collection Rhs>
    requires std::derived_from<Lhs, collection_view_interface<Lhs>> or std::derived_from<Rhs, collection_view_interface<Rhs>>
  constexpr std::partial_ordering
  operator<=>(const Lhs& lhs, const Rhs& rhs) noexcept
  {
    return detail::compare_impl(lhs, rhs);
  }

  template<collection Lhs, collection Rhs>
    requires std::derived_from<Lhs, collection_view_interface<Lhs>> or std::derived_from<Rhs, collection_view_interface<Rhs>>
  constexpr bool
  operator==(const Lhs& lhs, const Rhs& rhs) noexcept
  {
    return std::is_eq(operator<=>(lhs, rhs));
  }
#else
  namespace detail
  {
    template<std::size_t i = 0, typename T1, typename T2>
    constexpr int
    fixed_compare(const T1& lhs, const T2& rhs)
    {
      if constexpr (i == size_of_v<T1> and i == size_of_v<T2>) return 0;
      else if constexpr (i == size_of_v<T1>) return -1;
      else if constexpr (i == size_of_v<T2>) return +1;
      else
      {
        auto a = get(lhs, std::integral_constant<std::size_t, i>{});
        auto b = get(rhs, std::integral_constant<std::size_t, i>{});
        if (a == b) return fixed_compare<i + 1>(lhs, rhs);
        else return a == b ? 0 : a < b ? -1 : a > b ? +1 : +2;
      }
    }

    template<typename T1, typename T2>
    constexpr int
    compare_impl(const T1& lhs, const T2& rhs)
    {
      if constexpr (size_of_v<T1> != dynamic_size and size_of_v<T2> != dynamic_size) return fixed_compare(lhs, rhs);
      else
      {
#ifdef __cpp_lib_ranges
        namespace ranges = std::ranges;
#endif
        auto l = ranges::begin(lhs);
        auto r = ranges::begin(rhs);
        auto le = ranges::end(lhs);
        auto re = ranges::end(rhs);
        for (; l != le and r != re; ++l, ++r)
        {
          if (*l == *r) continue;
          if (*l < *r) return -1;
          if (*l > *r) return +1;
          return +2;
        }
        if (l == le and r == re) return 0;
        if (l == le) return -1;
        if (r == re) return +1;
        return +2;
      }
    }
  }


  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (std::is_base_of_v<collection_view_interface<Lhs>, Lhs> or std::is_base_of_v<collection_view_interface<Rhs>, Rhs>), int> = 0>
  constexpr bool operator==(const Lhs& lhs, const Rhs& rhs) noexcept { return detail::compare_impl(lhs, rhs) == 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (std::is_base_of_v<collection_view_interface<Lhs>, Lhs> or std::is_base_of_v<collection_view_interface<Rhs>, Rhs>), int> = 0>
  constexpr bool operator!=(const Lhs& lhs, const Rhs& rhs) noexcept { return detail::compare_impl(lhs, rhs) != 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (std::is_base_of_v<collection_view_interface<Lhs>, Lhs> or std::is_base_of_v<collection_view_interface<Rhs>, Rhs>), int> = 0>
  constexpr bool operator<(const Lhs& lhs, const Rhs& rhs) noexcept { return detail::compare_impl(lhs, rhs) < 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (std::is_base_of_v<collection_view_interface<Lhs>, Lhs> or std::is_base_of_v<collection_view_interface<Rhs>, Rhs>), int> = 0>
  constexpr bool operator>(const Lhs& lhs, const Rhs& rhs) noexcept { return detail::compare_impl(lhs, rhs) > 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (std::is_base_of_v<collection_view_interface<Lhs>, Lhs> or std::is_base_of_v<collection_view_interface<Rhs>, Rhs>), int> = 0>
  constexpr bool operator<=(const Lhs& lhs, const Rhs& rhs) noexcept { return detail::compare_impl(lhs, rhs) <= 0; }

  template<typename Lhs, typename Rhs, std::enable_if_t<collection<Lhs> and collection<Rhs> and
    (std::is_base_of_v<collection_view_interface<Lhs>, Lhs> or std::is_base_of_v<collection_view_interface<Rhs>, Rhs>), int> = 0>
  constexpr bool operator>=(const Lhs& lhs, const Rhs& rhs) noexcept { return detail::compare_impl(lhs, rhs) >= 0; }
#endif


} // namespace OpenKalman::collections


namespace std
{
  template<typename D>
  struct iterator_traits<OpenKalman::collections::internal::tuple_iterator<D>>
  {
    using difference_type = typename OpenKalman::collections::internal::tuple_iterator<D>::difference_type;
    using value_type = typename OpenKalman::collections::internal::tuple_iterator<D>::value_type;
    using iterator_category = typename OpenKalman::collections::internal::tuple_iterator<D>::iterator_category;
  };
}

#endif //OPENKALMAN_COLLECTION_VIEW_HPP
