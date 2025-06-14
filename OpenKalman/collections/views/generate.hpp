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
#include "values/concepts/size.hpp"
#include "collections/functions/compare.hpp"
#include "internal/movable_wrapper.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A \ref collection_view created by lazily generating elements based on an index.
   * \details Typically, the generating function will be a closure.
   * \tparam F A callable object (possibly a closure) taking an index and producing an output value corresponding to the index.
   * \tparam Size The size of the output collection. If it is <code>void</code>, the view is unsized.
   */
#ifdef __cpp_concepts
  template<typename F, values::size Size = std::unreachable_sentinel_t> requires std::same_as<Size, std::remove_reference_t<Size>> and
    std::invocable<F, std::size_t> and std::invocable<F, std::integral_constant<std::size_t, 0>>
  struct generate_view : std::ranges::view_interface<generate_view<F, Size>>
#else
  template<typename F, typename Size = unreachable_sentinel_t>
  struct generate_view : ranges::view_interface<generate_view<F, Size>>
#endif
  {
  private:

    using Size_ = std::conditional_t<values::index<Size>, Size, std::monostate>;

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
     * \brief Construct from a callable object and a size.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<const F, Arg&&> and values::index<Size>
    constexpr
    generate_view(Arg&& arg, Size_ size) noexcept
#else
    template<typename Arg, std::enable_if_t<std::is_constructible_v<const F, Arg&&> and values::index<Size>, int> = 0>
    constexpr
    generate_view(Arg&& arg, Size_ size) noexcept
#endif
      : f_box {const_cast<F&&>(std::forward<Arg>(arg))}, size_ {std::move(size)} {}


    /**
     * \brief Construct from a statically constructable callable object, if the view is unsized.
     */
#ifdef __cpp_concepts
    constexpr
    generate_view(const F&) noexcept requires static_F and (not values::index<Size>)
#else
    template<bool Enable = true, std::enable_if_t<Enable and static_F and (not values::index<Size>), int> = 0>
    constexpr
    generate_view(const F&) noexcept
#endif
      : f_box {F{}}, size_ {} {}


    /**
     * \brief Construct from a callable object, if the view is unsized.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::constructible_from<const F, Arg&&> and (not static_F) and (not values::index<Size>)
    constexpr
    generate_view(Arg&& arg) noexcept requires (not static_F)
#else
    template<bool Enable = true, typename Arg, std::enable_if_t<Enable and
      std::is_constructible_v<const F, Arg&&> and not static_F and (not values::index<Size>), int> = 0>
    constexpr
    generate_view(Arg&& arg) noexcept
#endif
      : f_box {const_cast<F&&>(std::forward<Arg>(arg))}, size_ {} {}


    /**
     * \brief Construct from a size if the function can be defined statically.
     */
#ifdef __cpp_concepts
    explicit constexpr
    generate_view(Size_ size) noexcept requires static_F and values::index<Size>
#else
    template<bool Enable = true, std::enable_if_t<Enable and static_F and values::index<Size>, int> = 0>
    explicit constexpr
    generate_view(Size_ size) noexcept
#endif
      : f_box {F{}}, size_ {std::move(size)} {}


    /**
     * \brief Default constructor.
     */
    constexpr
    generate_view() = default;


    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
    constexpr auto begin() { return iterator<false> {f_box, 0}; }

    /// \overload
    constexpr auto begin() const { return iterator<true> {f_box, 0}; }


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
    constexpr auto end()
    {
      using namespace std;
      if constexpr (values::index<Size>)
        return iterator<false> {f_box, static_cast<std::size_t>(size_)};
      else
        return unreachable_sentinel;
    }

    /// \overload
    constexpr auto end() const
    {
      using namespace std;
      if constexpr (values::index<Size>)
        return iterator<true> {f_box, static_cast<std::size_t>(size_)};
      else
        return unreachable_sentinel;
    }


    /**
     * \brief The size of the resulting object.
     */
#ifdef __cpp_concepts
    constexpr auto
    size() const noexcept requires values::index<Size>
#else
    template<bool Enable = true, std::enable_if_t<Enable and (values::index<Size>), int> = 0>
    constexpr auto size() const noexcept
#endif
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
      if constexpr (values::fixed<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return std::forward<decltype(self)>(self).f_box(std::integral_constant<std::size_t, i>{});
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      using namespace std;
      if constexpr (values::fixed<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      using namespace std;
      if constexpr (values::fixed<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      using namespace std;
      if constexpr (values::fixed<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return std::move(*this).f_box(std::integral_constant<std::size_t, i>{});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      using namespace std;
      if constexpr (values::fixed<Size>) static_assert(i < values::fixed_number_of_v<Size>, "Index out of range");
      return std::move(*this).f_box(std::integral_constant<std::size_t, i>{});
    }
#endif

  private:

    F_box f_box;
    Size_ size_;

  };


  /**
   * \brief Deduction guide
   */
  template<typename F, typename S>
  generate_view(F&&, const S&) -> generate_view<F, S>;

  /// \overload
  template<typename F>
  generate_view(F&&) -> generate_view<F>;


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
      /**
       * \brief Create a \ref generate_view.
       */
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


      /**
       * \brief Create an unsized \ref generate_view.
       */
#ifdef __cpp_concepts
      template<typename F> requires
        std::invocable<F, std::size_t> and std::invocable<F, std::integral_constant<std::size_t, 0>>
#else
      template<typename F, std::enable_if_t<std::is_invocable_v<F, std::size_t> and
        std::is_invocable_v<F, std::integral_constant<std::size_t, 0>>, int> = 0>
#endif
      constexpr auto
      operator() (F&& f) const
      {
        return generate_view {std::forward<F>(f)};
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
