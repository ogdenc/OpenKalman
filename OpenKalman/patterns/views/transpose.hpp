/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref collections::transpose_view and \ref collections::views::transpose.
 */

#ifndef OPENKALMAN_PATTERNS_VIEWS_TRANSPOSE_HPP
#define OPENKALMAN_PATTERNS_VIEWS_TRANSPOSE_HPP

#include <tuple>
#include <variant>
#include "collections/collections.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/functions/get_pattern.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief A view representing a transpose of a \ref pattern_collection.
   * \tparam indexa The first index to be swapped
   * \tparam indexb The second index to be swapped
   * \sa views::transpose, patterns::transpose
   */
#ifdef __cpp_lib_ranges
  template<pattern_collection P, std::size_t indexa = 0, std::size_t indexb = 1> requires (indexa < indexb)
#else
  template<typename P, std::size_t indexa = 0, std::size_t indexb = 1>
#endif
  struct transpose_view : stdex::ranges::view_interface<transpose_view<P, indexa, indexb>>
  {
  private:

    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

  public:

    /**
     * \brief Iterator for \ref transpose_view
     */
    template<bool Const>
    struct iterator
    {
      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = stdex::ranges::range_value_t<P>;
      using reference = stdex::ranges::range_reference_t<P>;
      using difference_type = std::ptrdiff_t;
      using pointer = void;

    private:

      using Parent = maybe_const<Const, transpose_view>;

    public:

      constexpr iterator() = default;

      constexpr iterator(Parent& parent, difference_type p) : parent_ {std::addressof(parent)}, current_{p} {}

      constexpr iterator(iterator<not Const> i) : parent_ {std::move(i.parent_)}, current_ {std::move(i.current_)} {}

      constexpr decltype(auto) operator*()
      {
        return get_pattern(parent_->p_.get(), current_ == indexa ? indexb : current_ == indexb ? indexa : current_);
      }

      constexpr decltype(auto) operator*() const
      {
        return get_pattern(parent_->p_.get(), current_ == indexa ? indexb : current_ == indexb ? indexa : current_);
      }

      constexpr decltype(auto) operator[](difference_type offset)
      {
        return get_pattern(parent_->p_.get(), (current_ == indexa ? indexb : current_ == indexb ? indexa : current_) + offset);
      }

      constexpr decltype(auto) operator[](difference_type offset) const
      {
        return get_pattern(parent_->p_.get(), (current_ == indexa ? indexb : current_ == indexb ? indexa : current_) + offset);
      }

      constexpr auto& operator++() noexcept { ++current_; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current_; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current_ += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current_ -= diff; return *this; }
      friend constexpr auto operator+(const iterator& it, const difference_type diff) noexcept
      { return iterator {*it.parent_, it.current_ + diff}; }
      friend constexpr auto operator+(const difference_type diff, const iterator& it) noexcept
      { return iterator {*it.parent_, diff + it.current_}; }
      friend constexpr auto operator-(const iterator& it, const difference_type diff)
      { return iterator {*it.parent_, it.current_ - diff}; }
      friend constexpr difference_type operator-(const iterator& it, const iterator& other) noexcept
      { return it.current_ - other.current_; }
      friend constexpr bool operator==(const iterator& it, const iterator& other) noexcept
      { return it.current_ == other.current_; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const iterator& other) const noexcept { return current_ <=> other.current_; }
#else
      constexpr bool operator!=(const iterator& other) const noexcept { return current_ != other.current_; }
      constexpr bool operator<(const iterator& other) const noexcept { return current_ < other.current_; }
      constexpr bool operator>(const iterator& other) const noexcept { return current_ > other.current_; }
      constexpr bool operator<=(const iterator& other) const noexcept { return current_ <= other.current_; }
      constexpr bool operator>=(const iterator& other) const noexcept { return current_ >= other.current_; }
#endif

    private:

      Parent * parent_;
      difference_type current_;

    };


    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr transpose_view() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<P>, int> = 0>
    constexpr transpose_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
    constexpr
    transpose_view(P&& p) : p_ {std::forward<P>(p)} {}


    /**
     * \brief The base view.
     **/
#ifdef __cpp_explicit_this_parameter
    constexpr decltype(auto)
    base(this auto&& self) noexcept { return std::forward<decltype(self)>(self).p_.get(); }
#else
    constexpr P& base() & { return this->p_.get(); }
    constexpr const P& base() const & { return this->p_.get(); }
    constexpr P&& base() && noexcept { return std::move(*this).p_.get(); }
    constexpr const P&& base() const && noexcept { return std::move(*this).p_.get(); }
#endif


    /**
     * \returns An iterator at the beginning, if the base object is a std::ranges::random_access_range.
     */
#ifdef __cpp_concepts
    constexpr auto
    begin() requires stdex::ranges::random_access_range<P>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::random_access_range<P>, int> = 0>
    constexpr auto begin()
#endif
    { return iterator<false> {*this, 0_uz}; }


#ifdef __cpp_concepts
    constexpr auto
    begin() const requires stdex::ranges::random_access_range<P>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::random_access_range<P>, int> = 0>
    constexpr auto begin() const
#endif
    { return iterator<true> {*this, 0_uz}; }


    /**
     * \returns An iterator at the end, if the base object is a std::ranges::random_access_range.
     */
#ifdef __cpp_concepts
    constexpr auto
    end() requires stdex::ranges::random_access_range<P>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::random_access_range<P>, int> = 0>
    constexpr auto
    end()
#endif
    {
      return iterator<false> {*this, static_cast<std::ptrdiff_t>(values::to_value_type(size()))};
    }


    /// \overload
#ifdef __cpp_concepts
    constexpr auto
    end() const requires stdex::ranges::random_access_range<P>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::ranges::random_access_range<P>, int> = 0>
    constexpr auto
    end() const
#endif
    {
      return iterator<true> {*this, static_cast<std::ptrdiff_t>(values::to_value_type(size()))};
    }


    /**
     * \returns The size of the object.
     */
#ifdef __cpp_concepts
    constexpr values::size auto
    size() const noexcept requires collections::sized<P>
#else
    template<bool Enable = true, std::enable_if_t<collections::sized<P>, int> = 0>
    constexpr auto
    size() const noexcept
#endif
    {
      struct Max { constexpr auto operator()(std::size_t a) const { return std::max(a, indexb + 1_uz); } };
      return values::operation(Max{}, get_size(p_.get()));
    }


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      constexpr std::size_t ti = i == indexa ? indexb : i == indexb ? indexa : i;
      return get_pattern<ti>(std::forward<decltype(self)>(self).p_.get());
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      constexpr std::size_t ti = i == indexa ? indexb : i == indexb ? indexa : i;
      return get_pattern<ti>(p_.get());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      constexpr std::size_t ti = i == indexa ? indexb : i == indexb ? indexa : i;
      return get_pattern<ti>(p_.get());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      constexpr std::size_t ti = i == indexa ? indexb : i == indexb ? indexa : i;
      return get_pattern<ti>(std::move(*this).p_.get());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      constexpr std::size_t ti = i == indexa ? indexb : i == indexb ? indexa : i;
      return get_pattern<ti>(std::move(*this).p_.get());
    }
#endif

  private:

    collections::internal::movable_wrapper<P> p_;

  };


  /**
   * \brief Deduction guides
   */
  template<typename P>
  transpose_view(P&&) -> transpose_view<P>;

}


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdex::ranges
#endif
{
  template<typename P, std::size_t indexa, std::size_t indexb>
  constexpr bool enable_borrowed_range<OpenKalman::patterns::transpose_view<P, indexa, indexb>> = enable_borrowed_range<P>;
}


#ifndef __cpp_lib_ranges
namespace OpenKalman::patterns::detail
{
  template<typename S, std::size_t indexb, typename = void>
  struct transpose_tuple_size {};

  template<typename S, std::size_t indexb>
  struct transpose_tuple_size<S, indexb, std::enable_if_t<
      values::fixed_value_compares_with<S, OpenKalman::stdex::dynamic_extent, &stdex::is_neq>>>
    : std::integral_constant<std::size_t, std::max(values::fixed_value_of_v<S>, indexb + 1_uz)> {};
}
#endif


namespace std
{
#ifdef __cpp_concepts
  template<typename P, std::size_t indexa, std::size_t indexb> requires
    OpenKalman::values::fixed_value_compares_with<OpenKalman::collections::size_of<P>, OpenKalman::stdex::dynamic_extent, &is_neq>
  struct tuple_size<OpenKalman::patterns::transpose_view<P, indexa, indexb>>
    : std::integral_constant<std::size_t, max(OpenKalman::collections::size_of_v<P>, indexb + 1)> {};
#else
  template<typename P, std::size_t indexa, std::size_t indexb>
  struct tuple_size<OpenKalman::patterns::transpose_view<P, indexa, indexb>>
    : OpenKalman::patterns::detail::transpose_tuple_size<OpenKalman::collections::size_of<P>, indexb> {};
#endif


  template<size_t i, typename P, std::size_t indexa, std::size_t indexb>
  struct tuple_element<i, OpenKalman::patterns::transpose_view<P, indexa, indexb>>
    : OpenKalman::patterns::pattern_collection_element<i == indexa ? indexb : i == indexb ? indexa : i, P> {};
}


namespace OpenKalman::patterns::views
{
  namespace detail
  {
    /**
     * \internal
     * \brief A closure for the transpose view
     */
#ifdef __cpp_concepts
    template<std::size_t indexa = 0, std::size_t indexb = 1> requires (indexa < indexb)
#else
    template<std::size_t indexa = 0, std::size_t indexb = 1>
#endif
    struct transpose_closure : stdex::ranges::range_adaptor_closure<transpose_closure<indexa, indexb>>
    {
      constexpr transpose_closure() = default;

#ifdef __cpp_concepts
      template<pattern_collection R>
#else
      template<typename R, std::enable_if_t<pattern_collection<R> and (indexa < indexb), int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const
      {
        if constexpr (values::fixed_value_compares_with<collections::size_of<R>, 0> or
          compares_with<pattern_collection_element_t<indexa, R>, pattern_collection_element_t<indexb, R>>)
        {
          if constexpr (collections::viewable_collection<R>) return collections::views::all(std::forward<R>(r));
          else return std::forward<R>(r);
        }
        else if constexpr (collections::viewable_collection<R>)
        {
          using AR = collections::views::all_t<R&&>;
          return transpose_view<AR, indexa, indexb> {collections::views::all(std::forward<R>(r))};
        }
        else
        {
          return transpose_view<R, indexa, indexb> {std::forward<R>(r)};
        }
      }
    };


#ifdef __cpp_concepts
    template<std::size_t indexa = 0, std::size_t indexb = 1> requires (indexa < indexb)
#else
    template<std::size_t indexa = 0, std::size_t indexb = 1>
#endif
    struct transpose_adapter
    {
      constexpr auto
      operator() () const
      {
        return transpose_closure<indexa, indexb> {};
      }


#ifdef __cpp_concepts
      template<pattern_collection R>
#else
      template<typename R, std::enable_if_t<pattern_collection<R>, int> = 0>
#endif
      constexpr decltype(auto)
      operator() (R&& r) const
      {
        return transpose_closure<indexa, indexb>{}(std::forward<R>(r));
      }

    };

  }


  /**
   * \brief a RangeAdapterObject associated with \ref transpose_view.
   * \details The expression <code>views::transpose(arg)</code> is expression-equivalent
   * to <code>transpose_view(arg)</code> for any suitable \ref collection arg.
   * \sa transpose_view
   */
#ifdef __cpp_concepts
  template<std::size_t indexa = 0, std::size_t indexb = 1> requires (indexa < indexb)
#else
  template<std::size_t indexa = 0, std::size_t indexb = 1, std::enable_if_t<indexa < indexb, int> = 0>
#endif
  inline constexpr detail::transpose_adapter<indexa, indexb> transpose;

}


#endif
