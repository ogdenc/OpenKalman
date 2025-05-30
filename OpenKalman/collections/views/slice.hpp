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
 * \brief Definition of \ref collections::slice_view and \ref collections::views::slice.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP

#include <tuple>
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/classes/operation.hpp"
#include "collections/concepts/sized_random_access_range.hpp"
#include "collections/concepts/collection.hpp"
#include "collections/concepts/viewable_collection.hpp"
#include "collections/functions/get.hpp"
#include "collections/functions/compare.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A view representing a slice of a \ref collection.
   * \details
   * The following should compile:
   * \code
   * \endcode
   * \tparam Offset The offset to the beginning of the slice
   * \tparam Extent The size of the slice
   * \sa views::slice
   */
#ifdef __cpp_lib_ranges
  template<collection V, values::index Offset, values::index Extent> requires
    std::same_as<std::decay_t<Offset>, Offset> and std::same_as<std::decay_t<Extent>, Extent> and
    (size_of_v<V> == dynamic_size or
    ((values::dynamic<Offset> or values::fixed_number_of_v<Offset> <= std::tuple_size_v<std::decay_t<V>>) and
    (values::dynamic<Extent> or values::fixed_number_of_v<Extent> <= std::tuple_size_v<std::decay_t<V>>) and
    (values::dynamic<Offset> or values::dynamic<Extent> or values::fixed_number_of_v<Offset> + values::fixed_number_of_v<Extent> <= std::tuple_size_v<std::decay_t<V>>)))
  struct slice_view : std::ranges::view_interface<slice_view<V, Offset, Extent>>
#else
  template<typename V, typename Offset, typename Extent>
  struct slice_view : ranges::view_interface<slice_view<V, Offset, Extent>>
#endif
  {
    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr slice_view() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and std::is_default_constructible_v<V> and
      std::is_default_constructible_v<Offset> and std::is_default_constructible_v<Extent>, int> = 0>
    constexpr slice_view() {}
#endif


    /**
     * \brief Construct from a \ref collection.
     */
    constexpr
    slice_view(const V& v, Offset offset, Extent extent)
      : v_ {v}, offset_ {std::move(offset)}, extent_ {std::move(extent)} {}

    /// \overload
    constexpr
    slice_view(V&& v, Offset offset, Extent extent)
      : v_ {std::move(v)}, offset_ {std::move(offset)}, extent_ {std::move(extent)} {}


    /**
     * \brief The base view.
     **/
#ifdef __cpp_explicit_this_parameter
    constexpr decltype(auto)
    base(this auto&& self) noexcept { return std::forward<decltype(self)>(self).v_; }
#else
    constexpr V& base() & { return this->v_; }
    constexpr const V& base() const & { return this->v_; }
    constexpr V&& base() && noexcept { return std::move(*this).v_; }
    constexpr const V&& base() const && noexcept { return std::move(*this).v_; }
#endif

  private:

    template<typename Self>
    static constexpr auto
    begin_impl(Self&& self) noexcept
    {
#ifdef __cpp_lib_ranges
      namespace ranges = std::ranges;
#else
      namespace ranges = OpenKalman::ranges;
#endif
      return ranges::begin(std::forward<Self>(self).v_);
    }

  public:

    /**
     * \returns An iterator at the beginning, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    begin(this auto&& self) noexcept
    {
      return begin_impl(std::forward<decltype(self)>(self)) + std::forward<decltype(self)>(self).offset_;
    }
#else
    constexpr auto begin() & { return begin_impl(*this) + offset_; }
    constexpr auto begin() const & { return begin_impl(*this) + offset_; }
    constexpr auto begin() && noexcept { return begin_impl(std::move(*this)) + std::move(*this).offset_; }
    constexpr auto begin() const && noexcept { return begin_impl(std::move(*this)) + std::move(*this).offset_; }
#endif


    /**
     * \returns An iterator at the end, if the base object is a range.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr auto
    end(this auto&& self) noexcept
    {
      return begin_impl(std::forward<decltype(self)>(self)) +
        std::forward<decltype(self)>(self).offset_ + std::forward<decltype(self)>(self).extent_;
    }
#else
    constexpr auto end() & { return begin_impl(*this) + offset_ + extent_; }
    constexpr auto end() const & { return begin_impl(*this) + offset_ + extent_; }
    constexpr auto end() && noexcept { return begin_impl(std::move(*this)) + std::move(*this).offset_ + std::move(*this).extent_; }
    constexpr auto end() const && noexcept { return begin_impl(std::move(*this)) + std::move(*this).offset_ + std::move(*this).extent_; }
#endif


    /**
     * \returns The size of the object.
     */
#ifdef __cpp_explicit_this_parameter
    constexpr values::index auto
    size(this auto&& self) noexcept { return std::forward<decltype(self)>(self).extent_; }
#else
    constexpr auto
    size() const noexcept { return extent_; }
#endif


    /**
     * \brief Get element i.
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(std::forward<decltype(self)>(self).v_,
        values::operation {std::plus{}, std::forward<decltype(self)>(self).offset_, std::integral_constant<std::size_t, i>{}});
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(v_,
        values::operation {std::plus{}, offset_, std::integral_constant<std::size_t, i>{}});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(v_,
        values::operation {std::plus{}, offset_, std::integral_constant<std::size_t, i>{}});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(std::move(*this).v_,
        values::operation {std::plus{}, offset_, std::integral_constant<std::size_t, i>{}});
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      if constexpr (values::fixed<Extent>) static_assert(i < values::fixed_number_of_v<Extent>, "Index exceeds range");
      return collections::get(std::move(*this).v_,
        values::operation {std::modulus{}, std::integral_constant<std::size_t, i>{}, get_size(std::move(*this).get_t())});
    }
#endif

  private:

    V v_;
    Offset offset_;
    Extent extent_;

  };


  /**
   * \brief Deduction guide
   */
  template<typename V, typename O, typename E>
  slice_view(const V&, const O&, const E&) -> slice_view<V, O, E>;

} // namespace OpenKalman


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename V, typename O, typename E>
  constexpr bool enable_borrowed_range<OpenKalman::collections::slice_view<V, O, E>> = enable_borrowed_range<V>;
}


#ifndef __cpp_lib_ranges
namespace OpenKalman::collections::detail
{
  template<std::size_t i, typename V, typename O, typename = void>
  struct slice_tuple_element
  {
    using type = ranges::range_value_t<V>;
  };

  template<std::size_t i, typename V, typename O>
  struct slice_tuple_element<i, V, O, std::enable_if_t<values::fixed<O>>>
    : std::tuple_element<values::fixed_number_of_v<O> + i, std::decay_t<V>> {};
}
#endif


namespace std
{
  template<typename V, typename O, typename E>
  struct tuple_size<OpenKalman::collections::slice_view<V, O, E>> : OpenKalman::values::fixed_number_of<E> {};


#ifdef __cpp_lib_ranges
  template<size_t i, typename V, OpenKalman::values::fixed O, typename E>
  struct tuple_element<i, OpenKalman::collections::slice_view<V, O, E>>
    : tuple_element<OpenKalman::values::fixed_number_of_v<O> + i, std::decay_t<V>> {};

  template<size_t i, typename V, typename O, typename E> requires (not OpenKalman::values::fixed<O>)
  struct tuple_element<i, OpenKalman::collections::slice_view<V, O, E>>
  {
    using type = ranges::range_value_t<V>;
  };
#else
  template<size_t i, typename V, typename O, typename E>
  struct tuple_element<i, OpenKalman::collections::slice_view<V, O, E>>
    : OpenKalman::collections::detail::slice_tuple_element<i, V, O> {};
#endif
} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    template<typename O, typename E>
    struct slice_closure
#if __cpp_lib_ranges >= 202202L
      : std::ranges::range_adaptor_closure<slice_closure<O, E>>
#else
      : ranges::range_adaptor_closure<slice_closure<O, E>>
#endif
    {
      constexpr slice_closure(O o, E e) : offset_ {std::move(o)}, extent_ {std::move(e)} {};

#ifdef __cpp_concepts
      template<viewable_collection R>
#else
      template<typename R, std::enable_if_t<viewable_collection<R>, int> = 0>
#endif
      constexpr auto
      operator() (R&& r) const
      {
        return slice_view {all(std::forward<R>(r)), offset_, extent_};
      }

    private:
      O offset_;
      E extent_;
    };


    struct slice_adapter
    {
#ifdef __cpp_concepts
      template<values::index O, values::index E>
#else
      template<typename O, typename E, std::enable_if_t<values::index<O> and values::index<E>, int> = 0>
#endif
      constexpr auto
      operator() (O o, E e) const
      {
        return slice_closure<O, E> {std::move(o), std::move(e)};
      }


#ifdef __cpp_concepts
      template<collection V, values::index O, values::index E>
#else
      template<typename V, typename O, typename E, std::enable_if_t<collection<V> and values::index<O> and values::index<E>, int> = 0>
#endif
      constexpr auto
      operator() (V&& t, O o, E e) const
      {
        return slice_view {all(std::forward<V>(t)), std::move(o), std::move(e)};
      }

    };

  }


  /**
   * \brief a RangeAdapterObject associated with \ref slice_view.
   * \details The expression <code>views::slice(arg)</code> is expression-equivalent
   * to <code>slice_view(arg)</code> for any suitable \ref collection arg.
   * \sa slice_view
   */
  inline constexpr detail::slice_adapter slice;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_SLICE_HPP
