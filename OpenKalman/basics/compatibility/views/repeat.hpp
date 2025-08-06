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
 * \brief Definition for \ref ranges::repeat_view and \ref ranges::views::repeat.
 */

#ifndef OPENKALMAN_COMPATIBILITY_VIEWS_REPEAT_HPP
#define OPENKALMAN_COMPATIBILITY_VIEWS_REPEAT_HPP

#include <type_traits>
#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/ranges.hpp"
#include "basics/compatibility/internal/movable_box.hpp"
#include "view_interface.hpp"

namespace OpenKalman::stdcompat::ranges
{
#ifdef __cpp_lib_ranges_repeat
  using std::ranges::repeat_view;
  namespace views
  {
    using std::ranges::views::repeat;
  }
#else
  /**
   * \brief Equivalent to std::ranges::repeat_view.
   */
#ifdef __cpp_lib_ranges
  template<std::move_constructible W, std::semiregular Bound = unreachable_sentinel_t> requires
    std::is_object_v<W> and std::same_as<W, std::remove_cv_t<W>> and
    (OpenKalman::internal::is_signed_integer_like<Bound> or
    (OpenKalman::internal::is_integer_like<Bound> and weakly_incrementable<Bound> or
    std::same_as<Bound, std::unreachable_sentinel_t>))
  struct repeat_view : std::ranges::view_interface<repeat_view<W, Bound>>
#else
  template<typename W, typename Bound = unreachable_sentinel_t>
  struct repeat_view : view_interface<repeat_view<W, Bound>>
#endif
  {
    struct iterator
    {
    private:

#ifdef __cpp_lib_ranges
      using index_type = std::conditional_t<std::is_same_v<Bound, std::unreachable_sentinel_t>, std::ptrdiff_t, Bound>;
#else
      using index_type = std::conditional_t<std::is_same_v<Bound, unreachable_sentinel_t>, std::ptrdiff_t, Bound>;
#endif

      template<typename I>
      using iota_diff_t = std::conditional_t<
        not std::is_integral_v<I> or (sizeof(iter_difference_t<I>) > sizeof(I)), iter_difference_t<I>, std::ptrdiff_t>;

    public:

      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = W;
      using difference_type = std::conditional_t<OpenKalman::internal::is_signed_integer_like<index_type>,
        index_type, iota_diff_t<index_type>>;
      using reference = W;
      using pointer = void;

      iterator() = default;

      constexpr explicit iterator(const W* value, index_type b = index_type{}) : value_ {value}, current_ {b} {}

      constexpr const W& operator*() const noexcept { return *value_; }

      constexpr const W& operator[](difference_type n) const noexcept { return *(*this + n); }

      constexpr iterator& operator++() { ++current_; return *this; }

      constexpr auto operator++(int) { auto tmp = *this; ++*this; return tmp; }

      constexpr iterator& operator--() { --current_; return *this; }

      constexpr auto operator--(int) { auto tmp = *this; --*this; return tmp; }

      constexpr iterator& operator+=(difference_type n) { current_ += n; return *this; }

      constexpr iterator& operator-=(difference_type n) { current_ -= n; return *this; }

      friend constexpr bool operator==(const iterator& x, const iterator& y) { return x.current_ == y.current_; }

#ifdef __cpp_impl_three_way_comparison
      friend constexpr auto operator<=>(const iterator& x, const iterator& y) { return x.current_ <=> y.current_; }
#else
      friend constexpr bool operator!=(const iterator& x, const iterator& y) { return x.current_ != y.current_; }
      friend constexpr bool operator<(const iterator& x, const iterator& y) { return x.current_ < y.current_; }
      friend constexpr bool operator>(const iterator& x, const iterator& y) { return x.current_ > y.current_; }
      friend constexpr bool operator<=(const iterator& x, const iterator& y) { return x.current_ <= y.current_; }
      friend constexpr bool operator>=(const iterator& x, const iterator& y) { return x.current_ >= y.current_; }
#endif

      friend constexpr auto operator+(iterator i, difference_type n) { i += n; return i; }

      friend constexpr auto operator+(difference_type n, iterator i) { i += n; return i; }

      friend constexpr auto operator-(iterator i, difference_type n) { i -= n; return i; }

      friend constexpr auto operator-(const iterator& x, const iterator& y)
      {
        return static_cast<difference_type>(x.current_) - static_cast<difference_type>(y.current_);
      }

    private:

      const W* value_;
      index_type current_;

    };


#ifdef __cpp_lib_concepts
    repeat_view() requires std::default_initializable<W> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<W>, int> = 0>
    constexpr repeat_view() {};
#endif


    constexpr explicit
    repeat_view(const W& value, Bound bound = {}) : value_ {value}, bound_ {bound} {}


    constexpr explicit
    repeat_view(W&& value, Bound bound = {}) : value_ {std::move(value)}, bound_ {bound} {}


#ifdef __cpp_lib_concepts
    template <typename...WArgs, typename...BoundArgs> requires
      std::constructible_from<W, WArgs...> and std::constructible_from<Bound, BoundArgs...>
#else
    template<typename...WArgs, typename...BoundArgs, std::enable_if_t<
      stdcompat::constructible_from<W, WArgs...> and stdcompat::constructible_from<Bound, BoundArgs...>, int> = 0>
#endif
    constexpr explicit
    repeat_view(std::piecewise_construct_t, std::tuple<WArgs...> value_args, std::tuple<BoundArgs...> bound_args = std::tuple<>{})
      : value_ {std::make_from_tuple<W>(std::move(value_args))}, bound_ {std::make_from_tuple<Bound>(std::move(bound_args))} {}


    constexpr iterator
    begin() const { return iterator {std::addressof(*value_)}; }


    template<bool Enable = true, std::enable_if_t<Enable and not std::is_same_v<Bound, unreachable_sentinel_t>, int> = 0>
    constexpr iterator
    end() const { return iterator {std::addressof(*value_), bound_}; }


    constexpr unreachable_sentinel_t
    end() const { return {}; }


#ifdef __cpp_lib_concepts
    constexpr auto
    size() const requires (not std::same_as<Bound, std::unreachable_sentinel_t>)
#else
    template<bool Enable = true, std::enable_if_t<Enable and not stdcompat::same_as<Bound, unreachable_sentinel_t>, int> = 0>
    constexpr auto size() const
#endif
    {
      if constexpr (std::is_integral_v<Bound>) return static_cast<std::make_unsigned_t<Bound>>(bound_);
      else return static_cast<std::size_t>(bound_);
    }

  private:

    OpenKalman::internal::movable_box<W> value_;

    Bound bound_;

  };


  template<typename W, typename Bound = unreachable_sentinel_t>
  repeat_view(W, Bound = {}) -> repeat_view<W, Bound>;


  namespace views
  {
    namespace detail
    {
      struct repeat_adapter
      {
  #ifdef __cpp_lib_concepts
        template<std::move_constructible W, std::semiregular Bound = unreachable_sentinel_t> requires
          (OpenKalman::internal::is_signed_integer_like<Bound> or
          (OpenKalman::internal::is_integer_like<Bound> and weakly_incrementable<Bound> or
          std::same_as<Bound, std::unreachable_sentinel_t>))
  #else
        template<typename W, typename Bound = unreachable_sentinel_t>
  #endif
        constexpr auto
        operator() [[nodiscard]] (W&& value, Bound&& bound = {}) const
        {
          return repeat_view<std::decay_t<W>, std::decay_t<Bound>> {std::forward<W>(value), std::forward<Bound>(bound)};
        }
      };
    }


    /**
     * \brief Equivalent to std::ranges::views::repeat.
     * \sa repeat_view
     */
    inline constexpr detail::repeat_adapter repeat;

  }

#endif
}


#endif
