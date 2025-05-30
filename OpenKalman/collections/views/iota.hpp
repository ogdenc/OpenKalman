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
 * \brief Definition for \ref collections::iota_view and \ref collections::views::iota.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_IOTA_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_IOTA_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "values/concepts/index.hpp"
#include "values/concepts/fixed.hpp"
#include "values/traits/fixed_number_of.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/functions/cast_to.hpp"
#include "collections/functions/compare.hpp"
#include "generate.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<values::integral Start = std::integral_constant<std::size_t, 0>>
#else
    template<typename Start = std::integral_constant<std::size_t, 0>, std::enable_if_t<values::integral<Start>, int> = 0>
#endif
    struct iota_generator
    {
      constexpr iota_generator() = default;

      explicit constexpr iota_generator(Start start) : start_ {std::move(start)} {};

#ifdef __cpp_concepts
      template<values::index I>
#else
      template<typename I, std::enable_if_t<values::index<I>, int> = 0>
#endif
      constexpr auto
      operator() (I i) const
      {
        return values::operation {std::plus<values::number_type_of_t<Start>>{}, start_, std::move(i)};
      }

    private:

      Start start_;
    };

  }


  /**
   * \brief An iota \ref collection that is a std::range and may also be \ref tuple_like.
   * \details In all cases, the result will be a std::range.
   * If the Size parameter is \ref values::fixed, then the result will also be
   * a \ref tuple_like sequence effectively in the form of
   * <code>std::integral_sequence<std::size_t, 0>{},...,std::integral_sequence<std::size_t, N>{}</code>
   * \tparam Start The start value of the iota.
   * \tparam Size The size of the resulting collection
   */
#ifdef __cpp_concepts
  template<values::integral Start, values::index Size = std::make_unsigned_t<values::number_type_of_t<Start>>> requires
    std::convertible_to<values::number_type_of_t<Size>, values::number_type_of_t<Start>> and
    std::same_as<Start, std::remove_reference_t<Start>> and std::same_as<Size, std::remove_reference_t<Size>>
#else
  template<typename Start, typename Size = std::make_unsigned_t<values::number_type_of_t<Start>>, typename = void>
#endif
  struct iota_view;


  /**
   * \brief Specialization in which Start is \ref values::dynamic
   */
#ifdef __cpp_concepts
  template<values::integral Start, values::index Size> requires (not values::fixed<Start>) and
    std::convertible_to<values::number_type_of_t<Size>, values::number_type_of_t<Start>> and
    std::same_as<Start, std::remove_reference_t<Start>> and std::same_as<Size, std::remove_reference_t<Size>>
  struct iota_view<Start, Size>
#else
  template<typename Start, typename Size>
  struct iota_view<Start, Size, std::enable_if_t<not values::fixed<Start>>>
#endif
    : generate_view<detail::iota_generator<Start>&, Size>
  {
  private:

    using base = generate_view<detail::iota_generator<Start>&, Size>;

  public:
    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr iota_view() requires values::fixed<Start> and values::fixed<Size>
#else
    template<bool Enable = true, std::enable_if_t<Enable and values::fixed<Start> and values::fixed<Size>, int> = 0>
    constexpr iota_view()
#endif
      : base {Size{}} {};


    /**
     * \brief Construct from an initial value and size.
     */
    constexpr iota_view(Start start, Size size)
      : base {gen_, std::move(size)}, gen_ {detail::iota_generator {std::move(start)}} {}


    /**
     * \brief Construct from a size, default-initializing Start.
     */
#ifdef __cpp_concepts
    explicit constexpr iota_view(Size size) requires values::fixed<Start>
#else
    template<bool Enable = true, std::enable_if_t<Enable and values::fixed<Start>>>
    explicit constexpr iota_view(Size size)
#endif
      : base {gen_, std::move(size)} {}

  private:

    detail::iota_generator<Start> gen_;

  };


  /**
   * \brief Specialization in which Start is \ref values::fixed
   */
  #ifdef __cpp_concepts
  template<values::integral Start, values::index Size> requires values::fixed<Start> and
    std::convertible_to<values::number_type_of_t<Size>, values::number_type_of_t<Start>> and
    std::same_as<Start, std::remove_reference_t<Start>> and std::same_as<Size, std::remove_reference_t<Size>>
  struct iota_view<Start, Size>
  #else
  template<typename Start, typename Size>
  struct iota_view<Start, Size, std::enable_if_t<values::fixed<Start>>>
  #endif
    : generate_view<detail::iota_generator<Start>, Size>
  {
  private:

    using base = generate_view<detail::iota_generator<Start>, Size>;

  public:
    /**
     * \brief Default constructor.
     */
  #ifdef __cpp_concepts
    constexpr iota_view() requires values::fixed<Start> and values::fixed<Size>
  #else
    template<bool Enable = true, std::enable_if_t<Enable and values::fixed<Start> and values::fixed<Size>, int> = 0>
    constexpr iota_view()
  #endif
    {};


    /**
     * \brief Construct from an initial value and size.
     */
    constexpr iota_view(Start, Size size) : base {std::move(size)} {}


    /**
     * \brief Construct from a size, default-initializing Start.
     */
  #ifdef __cpp_concepts
    explicit constexpr iota_view(Size size) requires values::fixed<Start>
  #else
    template<bool Enable = true, std::enable_if_t<Enable and values::fixed<Start>>>
    explicit constexpr iota_view(Size size)
  #endif
      : base {std::move(size)} {}

  };


  /**
   * \brief Deduction guide which assumes that omitting a Start parameter means that the sequence will start at zero.
   */
  template<typename Size>
  iota_view(const Size&) -> iota_view<std::integral_constant<values::number_type_of_t<Size>, 0>, Size>;

} // OpenKalman::values


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::ranges
#endif
{
  template<typename Start, typename Size>
  constexpr bool enable_borrowed_range<OpenKalman::collections::iota_view<Start, Size>> = true;
}


namespace std
{
#ifdef __cpp_concepts
  template<typename Start, OpenKalman::values::fixed Size>
  struct tuple_size<OpenKalman::collections::iota_view<Start, Size>>
    : std::integral_constant<size_t, OpenKalman::values::fixed_number_of_v<Size>> {};
#else
  template<typename Start, typename Size>
  struct tuple_size<OpenKalman::collections::iota_view<Start, Size>>
    : tuple_size<OpenKalman::collections::generate_view<OpenKalman::collections::detail::iota_generator<Start>, Size>> {};
#endif


#ifdef __cpp_concepts
  template<std::size_t i, OpenKalman::values::fixed Start, OpenKalman::values::fixed Size>
  struct tuple_element<i, OpenKalman::collections::iota_view<Start, Size>>
  {
    static_assert(i < OpenKalman::values::fixed_number_of_v<Size>);
    using type = OpenKalman::values::operation<std::plus<>, Start, std::integral_constant<std::size_t, i>>;
  };
#else
  template<std::size_t i, typename Start, typename Size>
  struct tuple_element<i, OpenKalman::collections::iota_view<Start, Size>>
    : tuple_element<i, OpenKalman::collections::generate_view<OpenKalman::collections::detail::iota_generator<Start>, Size>> {};
#endif

} // namespace std


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct iota_adapter
    {
#ifdef __cpp_concepts
      template<values::index Start, values::index Size>
#else
      template<typename Start, typename Size, std::enable_if_t<values::index<Start> and values::index<Size>, int> = 0>
#endif
      constexpr auto
      operator() (Start start, Size size) const
      {
        return iota_view(std::move(start), std::move(size));
      }


#ifdef __cpp_concepts
      template<values::index Size>
#else
      template<typename Size, std::enable_if_t<values::index<Size>, int> = 0>
#endif
      constexpr auto
      operator() (Size size) const
      {
        return iota_view(std::move(size));
      }
    };
  }


  /**
   * \brief a RangeAdapterObject associated with \ref iota_view.
   * \details The expression <code>views::iota(arg)</code> is expression-equivalent
   * to <code>iota_view(arg)</code> for any suitable \ref collection arg.
   * \sa iota_view
   */
  inline constexpr detail::iota_adapter iota;

}


#endif //OPENKALMAN_COLLECTIONS_VIEWS_IOTA_HPP
