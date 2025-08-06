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
 * \brief Definition for \ref collections::from_tuple_like.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_FROM_TUPLE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_FROM_TUPLE_HPP

#include "values/values.hpp"
#include "collections/functions/get.hpp"
#include "collections/functions/comparison_operators.hpp"
#include "internal/tuple_wrapper.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
    template<typename D, typename = std::make_index_sequence<std::tuple_size_v<std::decay_t<D>>>>
    struct tuple_iterator_call_table;


    template<typename D>
    struct tuple_iterator_call_table<D, std::index_sequence<>>
    {
      using element_type = std::size_t;

      static constexpr element_type
      call_table_get(const internal::tuple_wrapper<D>&) noexcept { return 0; }

      static constexpr std::array<decltype(call_table_get), 0>
      value {};
    };


    template<typename D, std::size_t...is>
    struct tuple_iterator_call_table<D, std::index_sequence<is...>>
    {
      using element_type = std::decay_t<common_tuple_type_t<D>>;

      template<std::size_t i>
      static constexpr auto
      call_table_get(const internal::tuple_wrapper<D>& box) noexcept
      {
        return static_cast<element_type>(OpenKalman::internal::generalized_std_get<i>(box));
      }

      static constexpr std::array
      value {call_table_get<is>...};
    };
  }


  /**
   * \brief A \ref collection_view created from a \ref viewable_tuple_like object.
   */
#ifdef __cpp_lib_ranges
  template<viewable_tuple_like Tup>
#else
  template<typename Tup>
#endif
  struct from_tuple_like : stdcompat::ranges::view_interface<from_tuple_like<Tup>>
  {
  private:

    using TupBox = internal::tuple_wrapper<Tup>;

    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

  public:

    /**
     * \brief Iterator for \ref from_tuple_like
     * \tparam Const Whether the iterator is constant
     */
    template<bool Const>
    struct iterator
    {
    private:

      using Parent = maybe_const<Const, from_tuple_like>;
      using call_table = detail::tuple_iterator_call_table<Tup>;

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
    from_tuple_like() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<TupBox>, int> = 0>
    constexpr
    from_tuple_like() {}
#endif


    /**
     * \brief Construct from a \ref viewable_tuple_like object.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<stdcompat::remove_cvref_t<Arg>, from_tuple_like>)
#else
    template<typename Arg, std::enable_if_t<(not std::is_same_v<stdcompat::remove_cvref_t<Arg>, from_tuple_like>), int> = 0>
#endif
    constexpr explicit 
    from_tuple_like(Arg&& arg) noexcept : tup_ {std::forward<Arg>(arg)} {}


    /**
     * \brief An iterator to the beginning of the tuple (treated as a std::ranges::range).
     * \note This incurs some overhead if the tuple-like object is not already a std::ranges::range,
     * because it must construct a call call_table to each element.
     */
    constexpr auto begin()
    {
      return iterator<false> {*this, 0_uz};
    }

    /// \overload
    constexpr auto begin() const
    {
      return iterator<true> {*this, 0_uz};
    }


    /**
     * \brief An iterator to the end of the tuple (treated as a std::ranges::range).
     * \note This incurs some overhead if the tuple-like object is not already a std::range,
     * because it must construct a call call_table to each element.
     */
    constexpr auto end()
    {
      return iterator<false> {*this, std::tuple_size_v<std::decay_t<Tup>>};
    }

    /// \overload
    constexpr auto end() const
    {
      return iterator<true> {*this, std::tuple_size_v<std::decay_t<Tup>>};
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
    front() { return OpenKalman::internal::generalized_std_get<0>(tup_); }

    /// \overload
    constexpr decltype(auto)
    front() const { return OpenKalman::internal::generalized_std_get<0>(tup_); }


    /**
     * \returns Returns the final element.
     */
    constexpr decltype(auto)
    back() { return OpenKalman::internal::generalized_std_get<size() - 1_uz>(tup_); }

    /// \overload
    constexpr decltype(auto)
    back() const { return OpenKalman::internal::generalized_std_get<size() - 1_uz>(tup_); }


    /**
     * \brief Subscript operator
     * \note Performs no runtime bounds checking
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
     * \note Performs no runtime bounds checking
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(i < std::tuple_size_v<std::decay_t<Tup>>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::forward<decltype(self)>(self).tup_);
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      static_assert(i < std::tuple_size_v<std::decay_t<Tup>>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(tup_);
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      static_assert(i < std::tuple_size_v<std::decay_t<Tup>>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(tup_);
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      static_assert(i < std::tuple_size_v<std::decay_t<Tup>>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).tup_);
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      static_assert(i < std::tuple_size_v<std::decay_t<Tup>>, "Index out of range");
      return OpenKalman::internal::generalized_std_get<i>(std::move(*this).tup_);
    }
#endif

  private:

    TupBox tup_;

  };


  template<typename Tup>
  from_tuple_like(Tup&&) -> from_tuple_like<Tup>;

} // namespace OpenKalman::collections


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdcompat::ranges
#endif
{
  template<typename Tup>
  constexpr bool enable_borrowed_range<OpenKalman::collections::from_tuple_like<Tup>> =
    std::is_lvalue_reference_v<Tup> or enable_borrowed_range<OpenKalman::stdcompat::remove_cvref_t<Tup>>;
}


namespace std
{
  template<typename Tup>
  struct tuple_size<OpenKalman::collections::from_tuple_like<Tup>> : tuple_size<decay_t<Tup>> {};


  template<std::size_t i, typename Tup>
  struct tuple_element<i, OpenKalman::collections::from_tuple_like<Tup>> : tuple_element<i, decay_t<Tup>> {};
}


#endif
