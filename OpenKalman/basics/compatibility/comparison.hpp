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
 * \brief Definitions relating to c+++20+ comparisons.
 */

#ifndef OPENKALMAN_COMPATIBILILTY_COMPARISON_HPP
#define OPENKALMAN_COMPATIBILILTY_COMPARISON_HPP

#include <utility>
#ifdef __cpp_impl_three_way_comparison
#include <compare>
#endif
#include "internal/exposition.hpp"
#include "common.hpp"

namespace OpenKalman::stdcompat
{
#ifdef __cpp_lib_integer_comparison_functions
  using std::cmp_equal;
  using std::cmp_not_equal;
  using std::cmp_less;
  using std::cmp_greater;
  using std::cmp_less_equal;
  using std::cmp_greater_equal;
#else
  template<typename T, typename U>
  constexpr bool cmp_equal(T t, U u) noexcept
  {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
    {
      return t == u;
    }
    else if constexpr (std::is_signed_v<T>)
    {
      return t >= 0 && std::make_unsigned_t<T>(t) == u;
    }
    else
    {
      return u >= 0 && std::make_unsigned_t<U>(u) == t;
    }
  }

  template<typename T, typename U>
  constexpr bool cmp_not_equal(T t, U u) noexcept
  {
    return not cmp_equal(t, u);
  }

  template<typename T, typename U>
  constexpr bool cmp_less(T t, U u) noexcept
  {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) return t < u;
    else if constexpr (std::is_signed_v<T>) return t < 0 or std::make_unsigned_t<T>(t) < u;
    else return u >= 0 and t < std::make_unsigned_t<U>(u);
  }

  template<typename T, typename U>
  constexpr bool cmp_greater(T t, U u) noexcept
  {
    return cmp_less(u, t);
  }

  template<typename T, typename U>
  constexpr bool cmp_less_equal(T t, U u) noexcept
  {
    return not cmp_less(u, t);
  }

  template<typename T, typename U>
  constexpr bool cmp_greater_equal(T t, U u) noexcept
  {
    return not cmp_less(t, u);
  }
#endif


#ifdef __cpp_impl_three_way_comparison
  using std::partial_ordering;
  using std::compare_three_way;
  using std::lexicographical_compare_three_way;
  using std::is_eq;
  using std::is_neq;
  using std::is_lt;
  using std::is_gt;
  using std::is_lteq;
  using std::is_gteq;
#else
  namespace detail
  {
    enum struct Ord : signed char { equivalent = 0, less = -1, greater = 1, unordered = 2 };
  }


  class partial_ordering
  {
    struct unspecified { constexpr unspecified(unspecified*) noexcept {} };

    detail::Ord my_value;

    explicit constexpr partial_ordering(detail::Ord value) noexcept : my_value(value) {}

  public:

    static const partial_ordering less;
    static const partial_ordering equivalent;
    static const partial_ordering greater;
    static const partial_ordering unordered;

    template<typename I, std::enable_if_t<stdcompat::constructible_from<std::ptrdiff_t, I>, int> = 0>
    explicit constexpr partial_ordering(I i) : my_value(static_cast<detail::Ord>(i)) {}

    [[nodiscard]] friend constexpr bool
    operator==(partial_ordering v, unspecified) noexcept { return v.my_value == detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator==(unspecified, partial_ordering v) noexcept { return v.my_value == detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator==(partial_ordering v, partial_ordering w) noexcept { return v.my_value == w.my_value; };

    [[nodiscard]] friend constexpr bool
    operator!=(partial_ordering v, unspecified) noexcept { return v.my_value != detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator!=(unspecified, partial_ordering v) noexcept { return v.my_value != detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator!=(partial_ordering v, partial_ordering w) noexcept { return v.my_value != w.my_value; };

    [[nodiscard]] friend constexpr bool
    operator<(partial_ordering v, unspecified) noexcept { return v.my_value == detail::Ord::less; }

    [[nodiscard]] friend constexpr bool
    operator<(unspecified, partial_ordering v) noexcept { return v.my_value == detail::Ord::greater; }

    [[nodiscard]] friend constexpr bool
    operator>(partial_ordering v, unspecified) noexcept { return v.my_value == detail::Ord::greater; }

    [[nodiscard]] friend constexpr bool
    operator>(unspecified, partial_ordering v) noexcept { return v.my_value == detail::Ord::less; }

    [[nodiscard]] friend constexpr bool
    operator<=(partial_ordering v, unspecified) noexcept { return v.my_value <= detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator<=(unspecified, partial_ordering v) noexcept { return v.my_value == detail::Ord::greater or v.my_value == detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator>=(partial_ordering v, unspecified) noexcept { return v.my_value == detail::Ord::greater or v.my_value == detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator>=(unspecified, partial_ordering v) noexcept { return v.my_value <= detail::Ord::equivalent; }
  };


  // valid values' definitions
  inline constexpr partial_ordering partial_ordering::less(detail::Ord::less);

  inline constexpr partial_ordering partial_ordering::equivalent(detail::Ord::equivalent);

  inline constexpr partial_ordering partial_ordering::greater(detail::Ord::greater);

  inline constexpr partial_ordering partial_ordering::unordered(detail::Ord::unordered);


  struct compare_three_way
  {
    template<typename T, typename U>
    constexpr partial_ordering
    operator() [[nodiscard]] (T&& t, U&& u) const
    {
      if (stdcompat::cmp_equal(t, u)) return partial_ordering::equivalent;
      if (stdcompat::cmp_less(t, u)) return partial_ordering::less;
      if (stdcompat::cmp_greater(t, u)) return partial_ordering::greater;
      return partial_ordering::unordered;
    }

    using is_transparent = void;
  };


  template<typename InputIt1, typename InputIt2, typename Cmp>
  constexpr auto
  lexicographical_compare_three_way(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, Cmp comp)
    -> decltype(comp(*first1, *first2))
  {
    using ret_t = decltype(comp(*first1, *first2));
    static_assert(std::disjunction_v<
                      //std::is_same<ret_t, strong_ordering>,
                      //std::is_same<ret_t, weak_ordering>,
                      std::is_same<ret_t, partial_ordering>>,
                  "The return type must be a comparison category type.");

    bool exhaust1 = (first1 == last1);
    bool exhaust2 = (first2 == last2);
    for (; not exhaust1 and not exhaust2; exhaust1 = (++first1 == last1), exhaust2 = (++first2 == last2))
      if (auto c = comp(*first1, *first2); c != partial_ordering(0)) return c;

    return not exhaust1 ? partial_ordering::greater: // strong_ordering::greater:
           not exhaust2 ? partial_ordering::less: // strong_ordering::less:
                          partial_ordering::equivalent; // strong_ordering::equal;
  }


  template<typename InputIt1, typename InputIt2>
  constexpr auto
  lexicographical_compare_three_way(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2 )
  {
    return lexicographical_compare_three_way(first1, last1, first2, last2, compare_three_way());
  }


  constexpr bool is_eq( partial_ordering cmp ) noexcept { return cmp == 0; };
  constexpr bool is_neq( partial_ordering cmp ) noexcept { return cmp != 0; };
  constexpr bool is_lt( partial_ordering cmp ) noexcept { return cmp < 0; };
  constexpr bool is_lteq( partial_ordering cmp ) noexcept { return cmp <= 0; };
  constexpr bool is_gt( partial_ordering cmp ) noexcept { return cmp > 0; };
  constexpr bool is_gteq( partial_ordering cmp ) noexcept { return cmp >= 0; };
#endif


#ifdef __cpp_lib_concepts
  using std::equality_comparable;
  using std::equality_comparable_with;
  using std::totally_ordered;
  using std::totally_ordered_with;
#else
  namespace detail
  {
    template<typename T, typename U, typename C = typename stdcompat::common_reference<const T&, const U&>::type, typename = void>
    struct ComparisonCommonTypeWithImpl : std::false_type {};

    template<typename T, typename U, typename C>
    struct ComparisonCommonTypeWithImpl<T, U, C, std::enable_if_t<
      stdcompat::same_as<common_reference_t<const T&, const U&>, common_reference_t<const U&, const T&>> and
      (stdcompat::convertible_to<const T&, const C&> or stdcompat::convertible_to<T&, const C&>) and
      (stdcompat::convertible_to<const U&, const C&> or stdcompat::convertible_to<U&, const C&>)
      >> : std::true_type {};


    template<typename T, typename U>
    inline constexpr bool
    ComparisonCommonTypeWith = ComparisonCommonTypeWithImpl<remove_cvref_t<T>, remove_cvref_t<U>>::value;
  }


  template<typename T>
  inline constexpr bool
  equality_comparable = OpenKalman::internal::WeaklyEqualityComparableWith<T, T>;


  template<typename T, typename U>
  inline constexpr bool
  equality_comparable_with =
    equality_comparable<T> and
    equality_comparable<U> and
    detail::ComparisonCommonTypeWith<T, U> and
    equality_comparable<common_reference_t<const std::remove_reference_t<T>&, const std::remove_reference_t<U>&>> and
    OpenKalman::internal::WeaklyEqualityComparableWith<T, U>;


  namespace detail
  {


  }


  template<typename T>
  inline constexpr bool
  totally_ordered = equality_comparable<T> and OpenKalman::internal::PartiallyOrderedWith<T, T>;


  namespace detail
  {
    template<typename T, typename U, typename = void>
    struct totally_ordered_with_impl : std::false_type {};

    template<typename T, typename U>
    struct totally_ordered_with_impl<T, U, std::enable_if_t<
      totally_ordered<typename stdcompat::common_reference<const std::remove_reference_t<T>&, const std::remove_reference_t<U>&>::type>>> : std::true_type {};
  }


  template<typename T, typename U>
  inline constexpr bool
  totally_ordered_with =
    totally_ordered<T> and
    totally_ordered<U> and
    equality_comparable_with<T, U> and
    detail::totally_ordered_with_impl<T, U>::value and
    OpenKalman::internal::PartiallyOrderedWith<T, U>;
#endif

}


#endif