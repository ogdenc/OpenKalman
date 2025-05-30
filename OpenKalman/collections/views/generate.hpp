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
 * \brief Definition for \ref collections::generate_view and \ref collections::views::generate.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_GENERATE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_GENERATE_HPP

#include <type_traits>
#include "basics/compatibility/language-features.hpp"
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "values/concepts/index.hpp"
#include "collections/functions/compare.hpp"
#include "internal/movable_wrapper.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A \ref collection_view created by lazily generating elements based on an index.
   * \details Typically, the generating function will be a closure.
   * \tparam F A callable object (possibly a closure) taking an index and producing an output value corresponding to the index.
   * \tparam Size The size of the output collection
   */
#ifdef __cpp_concepts
  template<typename F, values::index Size> requires std::same_as<Size, std::remove_reference_t<Size>> and
    std::invocable<F, std::size_t> and std::invocable<F, std::integral_constant<std::size_t, 0>>
  struct generate_view : std::ranges::view_interface<generate_view<F, Size>>
#else
  template<typename F, typename Size, typename = void>
  struct generate_view : ranges::view_interface<generate_view<F, Size>>
#endif
  {
  private:

    using F_box = internal::movable_wrapper<F>;

#if __cplusplus >= 202002L
    static constexpr bool static_F = std::semiregular<F_box>;
#else
    static constexpr bool static_F = semiregular<F_box>;
#endif


    template<bool Const, typename T>
    using maybe_const = std::conditional_t<Const, const T, T>;

  public:

    /**
     * \brief Iterator for \ref generate_view
     */
    template<bool Const>
    struct iterator
    {
    private:

      using F_box_c = maybe_const<Const, F_box>;

      using F_ref = std::conditional_t<static_F, F_box, F_box_c *>;

      constexpr F_box&
      get_f() const noexcept
      {
        if constexpr (static_F) return const_cast<F_box&>(f_ref);
        else return const_cast<F_box&>(*f_ref);
      }

      static constexpr F_ref
      init_f_ref(F_box_c& f) noexcept
      {
        if constexpr (static_F) return const_cast<F_ref&&>(f);
        else return std::addressof(f);
      }

    public:

      using iterator_concept = std::random_access_iterator_tag;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = std::invoke_result_t<F_box&, std::size_t>;
      using difference_type = std::ptrdiff_t;
      using reference = value_type;
      using pointer = void;

      constexpr iterator() = default;

      constexpr iterator(F_box_c& f, std::size_t pos)
        : f_ref {init_f_ref(f)}, current_ {static_cast<difference_type>(pos)} {};

      constexpr iterator(iterator<not Const> i) : f_ref {std::move(i.f_ref)}, current_ {std::move(i.current_)} {}

      constexpr value_type operator*() const
      {
        return get_f()(static_cast<std::size_t>(current_));
      }

      constexpr value_type operator[](difference_type offset) const
      {
        return get_f()(static_cast<std::size_t>(current_ + offset));
      }

      constexpr auto& operator++() noexcept { ++current_; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --current_; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { current_ += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { current_ -= diff; return *this; }

      friend constexpr auto operator+(const iterator& it, const difference_type diff)
      { return iterator {it.get_f(), static_cast<std::size_t>(it.current_ + diff)}; }
      friend constexpr auto operator+(const difference_type diff, const iterator& it)
      { return iterator {it.get_f(), static_cast<std::size_t>(diff + it.current_)}; }
      friend constexpr auto operator-(const iterator& it, const difference_type diff)
      { if (it.current_ < diff) throw std::out_of_range{"Iterator out of range"}; return iterator {it.get_f(), static_cast<std::size_t>(it.current_ - diff)}; }
      friend constexpr difference_type operator-(const iterator& it, const iterator& other)
      { return it.current_ - other.current_; }
      friend constexpr bool operator==(const iterator& it, const iterator& other)
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

      F_ref f_ref;
      difference_type current_;

    }; // struct Iterator


    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    generate_view() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      std::is_default_constructible_v<F_box> and std::is_default_constructible_v<Size>, int> = 0>
    constexpr generate_view() {}
#endif


    /**
     * \brief Construct from a size of the function can be defined statically.
     */
#ifdef __cpp_concepts
    explicit constexpr
    generate_view(Size size) noexcept requires static_F
#else
    template<bool Enable = static_F, std::enable_if_t<Enable, int> = 0>
    explicit constexpr
    generate_view(Size size) noexcept
#endif
      : size_ {std::move(size)} {}


    /**
     * \brief Construct from a callable object and a size, if the callable object can be defined statically.
     */
#ifdef __cpp_concepts
    constexpr
    generate_view(const F&, Size size) noexcept requires static_F
#else
    template<bool Enable = static_F, std::enable_if_t<Enable, int> = 0>
    constexpr
    generate_view(const F&, Size size) noexcept
#endif
      : size_ {std::move(size)} {}


    /**
     * \brief Construct from a callable object and a size.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<const F, Arg&&> and (not static_F)
    constexpr
    generate_view(Arg&& arg, Size size) noexcept requires (not static_F)
#else
    template<bool Enable = true, typename Arg, std::enable_if_t<Enable and
      std::is_constructible_v<const F, Arg&&> and not static_F, int> = 0>
    constexpr
    generate_view(Arg&& arg, Size size) noexcept
#endif
      : f_box {const_cast<F&&>(std::forward<Arg>(arg))}, size_ {std::move(size)} {}


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr auto begin() { return iterator<false> {f_box, 0}; }

    /// \overload
    constexpr auto begin() const { return iterator<true> {f_box, 0}; }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr auto end() { return iterator<false> {f_box, static_cast<std::size_t>(size_)}; }

    /// \overload
    constexpr auto end() const { return iterator<true> {f_box, static_cast<std::size_t>(size_)}; }

    /**
     * \brief The size of the resulting object.
     */
    constexpr auto
    size() const noexcept
    {
      return size_;
    }


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      if constexpr (values::dynamic<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return std::forward<decltype(self)>(self).f_box(std::integral_constant<std::size_t, i>{});
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      using namespace std;
      if constexpr (values::dynamic<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      using namespace std;
      if constexpr (values::dynamic<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      using namespace std;
      if constexpr (values::dynamic<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return std::move(*this).f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      using namespace std;
      if constexpr (values::dynamic<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return std::move(*this).f_box(std::integral_constant<std::size_t, i>{});
    }
#endif

  private:

    F_box f_box;
    Size size_;

  };


  template<typename F, typename S>
  generate_view(F&&, const S&) -> generate_view<F, S>;


} // namespace OpenKalman::values


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename F, typename S>
  constexpr bool enable_borrowed_range<OpenKalman::collections::generate_view<F, S>> = std::is_lvalue_reference_v<F> or
#if __cplusplus >= 202002L
    std::semiregular<OpenKalman::collections::internal::movable_wrapper<F>>;
#else
    semiregular<OpenKalman::collections::internal::movable_wrapper<F>>;
#endif
}


namespace std
{
  template<typename F, typename S>
  struct tuple_size<OpenKalman::collections::generate_view<F, S>> : OpenKalman::values::fixed_number_of<S> {};


  template<std::size_t i, typename F, typename S>
  struct tuple_element<i, OpenKalman::collections::generate_view<F, S>>
  {
    using type = std::invoke_result_t<F, std::integral_constant<std::size_t, i>>;
  };

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct generate_adaptor
    {
#ifdef __cpp_concepts
      template<typename F, values::index Size> requires
        std::invocable<F, std::size_t> and std::invocable<F, std::integral_constant<std::size_t, 0>>
#else
      template<typename F, typename Size, std::enable_if_t<values::index<Size> and
        std::is_invocable_v<F, std::size_t> and std::is_invocable_v<F, std::integral_constant<std::size_t, 0>>, int> = 0>
#endif
      constexpr auto
      operator() (F&& f, Size s) const
      {
        return generate_view {std::forward<F>(f), std::move(s)};
      }

    };

  }


  /**
   * \brief a \ref collection_view generator associated with \ref generate_view.
   * \details The expression <code>views::generate(f, s)</code> is expression-equivalent to <code>generate_view(f, s)</code>.
   * \sa generate_view
   */
  inline constexpr detail::generate_adaptor generate;

}

#endif //OPENKALMAN_COLLECTIONS_VIEWS_GENERATE_HPP
