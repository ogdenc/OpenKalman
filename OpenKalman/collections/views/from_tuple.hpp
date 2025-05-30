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
 * \brief Definition for \ref collections::from_tuple.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_FROM_TUPLE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_FROM_TUPLE_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#include "basics/compatibility/views/view_interface.hpp"
#endif
#include "basics/compatibility/language-features.hpp"
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "collections/functions/get.hpp"
#include "collections/functions/compare.hpp"
#include "internal/movable_wrapper.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename D, typename = std::make_index_sequence<std::tuple_size_v<D>>>
    struct tuple_iterator_call_table;


    template<typename D>
    struct tuple_iterator_call_table<D, std::index_sequence<>>
    {
      using element_type = int;

      static constexpr int
      call_table_get(const D&) noexcept { return 0; }

      static constexpr std::array<decltype(call_table_get), 0> value {};
    };


    template<typename D, std::size_t...is>
    struct tuple_iterator_call_table<D, std::index_sequence<is...>>
    {
      using element_type = common_tuple_type_t<D>;

      template<std::size_t i>
      static constexpr element_type
      call_table_get(const D& tup) noexcept { return OpenKalman::internal::generalized_std_get<i>(tup); }

      static constexpr std::array value {call_table_get<is>...};
    };
  }


  /**
   * \brief A \ref collection_view created from a \ref uniform_tuple_like object.
   */
#ifdef __cpp_lib_ranges
  template<uniform_tuple_like Tup>
  struct from_tuple : std::ranges::view_interface<from_tuple<Tup>>
#else
  template<typename Tup>
  struct from_tuple : ranges::view_interface<from_tuple<Tup>>
#endif
  {
  private:

    using TupBox = internal::movable_wrapper<Tup>;

    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

  public:

    /**
     * \brief Iterator for \ref from_tuple
     * \tparam Const Whether the iterator is constant
     */
    template<bool Const>
    struct iterator
    {
    private:

      using Parent = maybe_const<Const, from_tuple>;
      using call_table = detail::tuple_iterator_call_table<std::decay_t<TupBox>>;

    public:

      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = maybe_const<Const, typename call_table::element_type>;
      using difference_type = std::ptrdiff_t;
      using reference = typename call_table::element_type&;
      using pointer = void;
      constexpr iterator() = default;
      constexpr iterator(Parent& p, std::size_t pos) : parent_ {std::addressof(p)}, current_{static_cast<difference_type>(pos)} {}
      constexpr iterator(iterator<not Const> it) : parent_ {std::move(it.parent_)}, current_ {std::move(it.current_)} {}
      constexpr decltype(auto) operator*() { return call_table::value[current_](parent_->tup_); }
      constexpr decltype(auto) operator*() const { return call_table::value[current_](parent_->tup_); }
      constexpr decltype(auto) operator[](difference_type offset) { return call_table::value[current_ + offset](parent_->tup_); }
      constexpr decltype(auto) operator[](difference_type offset) const { return call_table::value[current_ + offset](parent_->tup_); }
      constexpr auto& operator++() { ++current_; return *this; }
      constexpr auto operator++(int) { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() { --current_; return *this; }
      constexpr auto operator--(int) { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) { current_ += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) { current_ -= diff; return *this; }
      friend constexpr auto operator+(const iterator& it, const difference_type diff)
        { return iterator {*it.parent_, static_cast<std::size_t>(it.current_ + diff)}; }
      friend constexpr auto operator+(const difference_type diff, const iterator& it)
        { return iterator {*it.parent_, static_cast<std::size_t>(diff + it.current_)}; }
      friend constexpr auto operator-(const iterator& it, const difference_type diff)
        { if (it.current_ < diff) throw std::out_of_range{"Iterator out of range"}; return iterator {*it.parent_, static_cast<std::size_t>(it.current_ - diff)}; }
      friend constexpr difference_type operator-(const iterator& it, const iterator& other)
        { return it.current_ - other.current_; }
      friend constexpr bool operator==(const iterator& it, const iterator& other)
        { return it.current_ == other.current_; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const iterator& other) const { return current_ <=> other.current_; }
#else
      constexpr bool operator!=(const iterator& other) const { return current_ != other.current_; }
      constexpr bool operator<(const iterator& other) const { return current_ < other.current_; }
      constexpr bool operator>(const iterator& other) const { return current_ > other.current_; }
      constexpr bool operator<=(const iterator& other) const { return current_ <= other.current_; }
      constexpr bool operator>=(const iterator& other) const { return current_ >= other.current_; }
#endif

    private:

      Parent * parent_;
      difference_type current_;

    }; // struct iterator


    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    from_tuple() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<TupBox>, int> = 0>
    constexpr
    from_tuple() {}
#endif


    /**
     * \brief Construct from a \ref uniform_tuple_like object.
     */
#if defined(__cpp_concepts) and defined(__cpp_lib_remove_cvref)
    template<typename Arg> requires std::constructible_from<TupBox, Arg&&> and (not std::same_as<std::remove_cvref_t<Arg>, from_tuple>)
#else
    template<typename Arg, std::enable_if_t<
      std::is_constructible_v<TupBox, Arg&&> and (not std::is_same_v<remove_cvref_t<Arg>, from_tuple>), int> = 0>
#endif
    constexpr explicit 
    from_tuple(Arg&& arg) noexcept : tup_ {std::forward<Arg>(arg)} {}


    /**
     * \brief The base tuple.
     **/
#ifdef __cpp_explicit_this_parameter
    constexpr decltype(auto)
    base(this auto&& self) noexcept { return std::forward<decltype(self)>(self).tup_.get(); }
#else
    constexpr decltype(auto) base() & { return this->tup_.get(); }
    constexpr decltype(auto) base() const & { return this->tup_.get(); }
    constexpr decltype(auto) base() && noexcept { return std::move(*this).tup_.get(); }
    constexpr decltype(auto) base() const && noexcept { return std::move(*this).tup_.get(); }
#endif


    /**
     * \brief An iterator to the beginning of the tuple (treated as a std::ranges::range).
     * \note This incurs some overhead if the tuple-like object is not already a std::ranges::range,
     * because it must construct a call call_table to each element.
     */
    constexpr auto begin()
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      if constexpr (ranges::range<Tup>) return ranges::begin(base());
      else return iterator<false> {*this, 0_uz};
    }

    /// \overload
    constexpr auto begin() const
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      if constexpr (ranges::range<Tup>) return ranges::begin(base());
      else return iterator<true> {*this, 0_uz};
    }


    /**
     * \brief An iterator to the end of the tuple (treated as a std::ranges::range).
     * \note This incurs some overhead if the tuple-like object is not already a std::range,
     * because it must construct a call call_table to each element.
     */
    constexpr auto end()
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      if constexpr (ranges::range<Tup>) return ranges::end(base());
      else return iterator<false> {*this, std::tuple_size_v<std::decay_t<Tup>>};
    }

    /// \overload
    constexpr auto end() const
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#endif
      if constexpr (ranges::range<Tup>) return ranges::end(base());
      else return iterator<true> {*this, std::tuple_size_v<std::decay_t<Tup>>};
    }


    /**
     * \brief The size of the resulting object.
     */
    static constexpr auto
    size() { return std::tuple_size<std::decay_t<Tup>>{}; }


    /**
     * \returns Indicates whether the base object is empty.
     */
    static constexpr auto
    empty()
    {
      if constexpr (std::tuple_size_v<std::decay_t<Tup>> == 0) return std::true_type{};
      else return std::false_type{};
    }


    /**
     * \returns Returns the initial element.
     */
    constexpr decltype(auto)
    front() { return OpenKalman::internal::generalized_std_get<0>(base()); }

    /// \overload
    constexpr decltype(auto)
    front() const { return OpenKalman::internal::generalized_std_get<0>(base()); }


    /**
     * \returns Returns the final element.
     */
    constexpr decltype(auto)
    back() { return OpenKalman::internal::generalized_std_get<size() - 1_uz>(base()); }

    /// \overload
    constexpr decltype(auto)
    back() const { return OpenKalman::internal::generalized_std_get<size() - 1_uz>(base()); }


    /**
     * \brief Subscript operator
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self, values::index I>
    constexpr decltype(auto)
    operator[](this Self&& self, I i) noexcept
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_number_of<I>::value < size(), "Index out of range");
      return collections::get(std::forward<Self>(self), std::move(i));
    }
#else
    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_number_of<I>::value < size(), "Index out of range");
      return collections::get(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_number_of<I>::value < size(), "Index out of range");
      return collections::get(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) && noexcept
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_number_of<I>::value < size(), "Index out of range");
      return collections::get(std::move(*this), std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const && noexcept
    {
      if constexpr (values::fixed<I>) static_assert(values::fixed_number_of<I>::value < size(), "Index out of range");
      return collections::get(std::move(*this), std::move(i));
    }
#endif


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(i < size(), "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::forward<decltype(self)>(self).base());
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      static_assert(i < size(), "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      static_assert(i < size(), "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert(i < size(), "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert(i < size(), "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).base());
    }
#endif

  private:

    TupBox tup_;

  };


  template<typename Tup>
  from_tuple(Tup&&) -> from_tuple<Tup>;

} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename Tup>
  constexpr bool enable_borrowed_range<OpenKalman::collections::from_tuple<Tup>> =
    std::is_lvalue_reference_v<Tup> or enable_borrowed_range<remove_cvref_t<Tup>>;
}


namespace std
{
  template<typename Tup>
  struct tuple_size<OpenKalman::collections::from_tuple<Tup>> : tuple_size<decay_t<Tup>> {};


  template<std::size_t i, typename Tup>
  struct tuple_element<i, OpenKalman::collections::from_tuple<Tup>> : tuple_element<i, decay_t<Tup>> {};
}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_FROM_TUPLE_HPP
