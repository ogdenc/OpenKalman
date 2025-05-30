/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions relating to the availability of c++ language features.
 */

#ifndef OPENKALMAN_LANGUAGE_FEATURES_HPP
#define OPENKALMAN_LANGUAGE_FEATURES_HPP

#ifdef __clang__
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS   true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2 (__clang_major__ >= 15) // optimal value may be as low as > 10 (ver. 10.0.0)
#elif defined(__GNUC__)
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS   (__GNUC__ >= 20) // optimal value may be as low as > 10 (ver. 10.1.0)
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2 (__GNUC__ >= 12) // optimal value may be as low as > 10 (ver. 10.1.0)
#else
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS   true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2 true
#endif


#ifdef __cpp_lib_math_constants
#include <numbers>
namespace OpenKalman::numbers { using namespace std::numbers; }
#else
// These re-create the c++20 mathematical constants.
namespace OpenKalman::numbers
{
#ifdef __cpp_lib_concepts
#include <concepts>
  template<std::floating_point T> inline constexpr T e_v = 2.718281828459045235360287471352662498L;
  template<std::floating_point T> inline constexpr T log2e_v = 1.442695040888963407359924681001892137L;
  template<std::floating_point T> inline constexpr T log10e_v = 0.434294481903251827651128918916605082L;
  template<std::floating_point T> inline constexpr T pi_v = 3.141592653589793238462643383279502884L;
  template<std::floating_point T> inline constexpr T inv_pi_v = 0.318309886183790671537767526745028724L;
  template<std::floating_point T> inline constexpr T inv_sqrtpi_v = 0.564189583547756286948079451560772586L;
  template<std::floating_point T> inline constexpr T ln2_v = 0.693147180559945309417232121458176568L;
  template<std::floating_point T> inline constexpr T ln10_v = 2.302585092994045684017991454684364208L;
  template<std::floating_point T> inline constexpr T sqrt2_v = 1.414213562373095048801688724209698079L;
  template<std::floating_point T> inline constexpr T sqrt3_v = 1.732050807568877293527446341505872367L;
  template<std::floating_point T> inline constexpr T inv_sqrt3_v = 0.577350269189625764509148780501957456L;
  template<std::floating_point T> inline constexpr T egamma_v = 0.577215664901532860606512090082402431L;
  template<std::floating_point T> inline constexpr T phi_v = 1.618033988749894848204586834365638118L;
#else
#include <type_traits>
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T e_v = 2.718281828459045235360287471352662498L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T log2e_v = 1.442695040888963407359924681001892137L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T log10e_v = 0.434294481903251827651128918916605082L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T pi_v = 3.141592653589793238462643383279502884L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T inv_pi_v = 0.318309886183790671537767526745028724L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T inv_sqrtpi_v = 0.564189583547756286948079451560772586L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T ln2_v = 0.693147180559945309417232121458176568L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T ln10_v = 2.302585092994045684017991454684364208L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T sqrt2_v = 1.414213562373095048801688724209698079L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T sqrt3_v = 1.732050807568877293527446341505872367L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T inv_sqrt3_v = 0.577350269189625764509148780501957456L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T egamma_v = 0.577215664901532860606512090082402431L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T phi_v = 1.618033988749894848204586834365638118L;
#endif

  inline constexpr double e = e_v<double>;
  inline constexpr double log2e = log2e_v<double>;
  inline constexpr double log10e = log10e_v<double>;
  inline constexpr double pi = pi_v<double>;
  inline constexpr double inv_pi = inv_pi_v<double>;
  inline constexpr double inv_sqrtpi = inv_sqrtpi_v<double>;
  inline constexpr double ln2 = ln2_v<double>;
  inline constexpr double ln10 = ln10_v<double>;
  inline constexpr double sqrt2 = sqrt2_v<double>;
  inline constexpr double sqrt3 = sqrt3_v<double>;
  inline constexpr double inv_sqrt3 = inv_sqrt3_v<double>;
  inline constexpr double egamma = egamma_v<double>;
  inline constexpr double phi = phi_v<double>;
}
#endif


// std::size_t literal similar and equivalent to "uz" literal defined in c++23 standard.
constexpr std::size_t operator ""_uz(unsigned long long x) { return x; };


#ifndef __cpp_lib_integer_comparison_functions
namespace OpenKalman
{
  template<typename T, typename U>
  constexpr bool cmp_equal(T t, U u) noexcept
  {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
      return t == u;
    else if constexpr (std::is_signed_v<T>)
      return t >= 0 && std::make_unsigned_t<T>(t) == u;
    else
      return u >= 0 && std::make_unsigned_t<U>(u) == t;
  }

  template<typename T, typename U>
  constexpr bool cmp_not_equal(T t, U u) noexcept
  {
    return !cmp_equal(t, u);
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
}
#endif


#ifndef __cpp_impl_three_way_comparison
namespace OpenKalman
{
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

    template<typename I, std::enable_if_t<std::is_constructible_v<std::ptrdiff_t, I>, int> = 0>
    explicit constexpr partial_ordering(I i) : my_value(static_cast<detail::Ord>(i)) {}

    [[nodiscard]] friend constexpr bool
    operator==(partial_ordering v, unspecified) noexcept { return v.my_value == detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator==(unspecified, partial_ordering v) noexcept { return v.my_value == detail::Ord::equivalent; }

    [[nodiscard]] friend constexpr bool
    operator==(partial_ordering v, partial_ordering w) noexcept { return v.my_value == w.my_value; };

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
      if (t == u) return partial_ordering::equivalent;
      if (t < u) return partial_ordering::less;
      if (t > u) return partial_ordering::greater;
      return partial_ordering::unordered;
    }

    using is_transparent = void;
  };

}
#endif


#if not defined(__cpp_lib_remove_cvref) or not defined(__cpp_lib_ranges)
namespace OpenKalman
{
  template<typename T>
  struct remove_cvref { using type = std::remove_cv_t<std::remove_reference_t<T>>; };

  template<typename T>
  using remove_cvref_t = typename remove_cvref<T>::type;
}
#endif


namespace OpenKalman::internal
{
  namespace detail
  {
    struct decay_copy_impl final
    {
      template<typename T>
      constexpr std::decay_t<T> operator()(T&& t) const noexcept { return std::forward<T>(t); }
    };
  }

  inline constexpr detail::decay_copy_impl decay_copy;
}


#ifndef __cpp_lib_bounded_array_traits
namespace OpenKalman
{
  template<typename T>
  struct is_bounded_array : std::false_type {};

  template<typename T, std::size_t N>
  struct is_bounded_array<T[N]> : std::true_type {};

  template<typename T>
  constexpr bool is_bounded_array_v = is_bounded_array<T>::value;
}
#endif


#if __cplusplus < 202002L
namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename T, typename = void>
    struct is_integer_like_impl : std::false_type {};

    template<>
    struct is_integer_like_impl<bool> : std::false_type {};

    template<>
    struct is_integer_like_impl<const bool> : std::false_type {};

    template<>
    struct is_integer_like_impl<volatile bool> : std::false_type {};

    template<>
    struct is_integer_like_impl<const volatile bool> : std::false_type {};

    template<typename T>
    struct is_integer_like_impl<T, std::enable_if_t<std::is_integral_v<T>>> : std::true_type {};
  }

  template<typename T>
  inline constexpr bool is_integer_like = detail::is_integer_like_impl<T>::value;

  template<typename T>
  inline constexpr bool is_signed_integer_like = is_integer_like<T> and std::is_signed_v<T>;

  template<typename T>
  inline constexpr bool is_unsigned_integer_like = is_integer_like<T> and std::is_unsigned_v<T>;

}


namespace OpenKalman
{
  struct identity
  {
    template<typename T>
    [[nodiscard]] constexpr T&& operator()(T&& t) const noexcept { return std::forward<T>(t); }

    struct is_transparent; // undefined
  };


  template<typename T>
  struct type_identity { using type = T; };

  template<typename T>
  using type_identity_t = typename type_identity<T>::type;


  /**
   * \internal
   * \brief A constexpr version of std::reference_wrapper, for use when compiling in c++17
   **/
  namespace detail
  {
    template<typename T> constexpr T& reference_wrapper_FUN(T& t) noexcept { return t; }
    template<typename T> void reference_wrapper_FUN(T&&) = delete;
  }


  template<typename T>
  class reference_wrapper
  {
#ifdef __cpp_lib_remove_cvref
    using std::remove_cvref_t;
#endif

  public:

    using type = T;

    template<typename U, typename = std::void_t<decltype(detail::reference_wrapper_FUN<T>(std::declval<U>()))>,
      std::enable_if_t<not std::is_same_v<reference_wrapper, remove_cvref_t<U>>, int> = 0>
    constexpr reference_wrapper(U&& u) noexcept(noexcept(detail::reference_wrapper_FUN<T>(std::forward<U>(u))))
      : ptr(std::addressof(detail::reference_wrapper_FUN<T>(std::forward<U>(u)))) {}


    reference_wrapper(const reference_wrapper&) noexcept = default;


    reference_wrapper& operator=(const reference_wrapper& x) noexcept = default;


    constexpr operator T& () const noexcept { return *ptr; }
    constexpr T& get() const noexcept { return *ptr; }


    template<typename... ArgTypes>
    constexpr std::invoke_result_t<T&, ArgTypes...>
    operator() (ArgTypes&&... args ) const noexcept(std::is_nothrow_invocable_v<T&, ArgTypes...>)
    {
      return get()(std::forward<ArgTypes>(args)...);
    }

  private:

    T* ptr;

  };

  // deduction guides
  template<typename T>
  reference_wrapper(T&) -> reference_wrapper<T>;


  template<typename T>
  constexpr std::reference_wrapper<T>
  ref(T& t) noexcept { return {t}; };

  template<typename T>
  constexpr std::reference_wrapper<T>
  ref(std::reference_wrapper<T> t) noexcept { return std::move(t); };

  template<typename T>
  void ref(const T&&) = delete;

  template<typename T>
  constexpr std::reference_wrapper<const T>
  cref(const T& t) noexcept { return {t}; };

  template<typename T>
  constexpr std::reference_wrapper<const T>
  cref(std::reference_wrapper<T> t) noexcept { return std::move(t); };

  template<typename T>
  void cref(const T&&) = delete;


  // ---
  // copyable, movable, semiregular
  // ---

  template<typename T>
  inline constexpr bool movable =
    std::is_object_v<T> and
    std::is_move_constructible_v<T> and
    std::is_assignable_v<T&, T> /*and
    std::swappable<T>*/;


  template<typename T>
  inline constexpr bool copyable =
    std::is_copy_constructible_v<T> and
    movable<T> and
    std::is_assignable_v<T&, T&> and
    std::is_assignable_v<T&, const T&> and
    std::is_assignable_v<T&, const T>;


  template<typename T>
  inline constexpr bool semiregular = copyable<T> and std::is_default_constructible_v<T>;

}
#endif


// ---
// invoke, invoke_r
// ---

#if __cplusplus < 202002L
namespace OpenKalman
{
  namespace detail
  {
    template<typename> static constexpr bool is_reference_wrapper_v = false;
    template<typename U> static constexpr bool is_reference_wrapper_v<std::reference_wrapper<U>> = true;

    template<typename T> using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template<typename C, typename P, typename O, typename...Args>
    static constexpr decltype(auto)
    invoke_memptr(P C::* member, O&& object, Args&&... args)
    {
      if constexpr (std::is_function_v<P>)
      {
        if constexpr (std::is_same_v<C, remove_cvref_t<O>> or std::is_base_of_v<C, remove_cvref_t<O>>)
          return (std::forward<O>(object) .* member)(std::forward<Args>(args)...);
        else if constexpr (is_reference_wrapper_v<remove_cvref_t<O>>)
          return (object.get() .* member)(std::forward<Args>(args)...);
        else
          return ((*std::forward<O>(object)) .* member)(std::forward<Args>(args)...);
      }
      else
      {
        static_assert(std::is_object_v<P> && sizeof...(args) == 0);
        if constexpr (std::is_same_v<C, remove_cvref_t<O>> or std::is_base_of_v<C, remove_cvref_t<O>>)
          return std::forward<O>(object) .* member;
        else if constexpr (is_reference_wrapper_v<remove_cvref_t<O>>)
          return object.get() .* member;
        else
          return (*std::forward<O>(object)) .* member;
      }
    }
  }


  /**
   * \internal
   * \brief A constexpr version of std::invoke, for use when compiling in c++17
   **/
  template<typename F, typename...Args>
  static constexpr decltype(auto)
  invoke(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable_v<F, Args...>)
  {
    if constexpr (std::is_member_pointer_v<remove_cvref_t<F>>)
      return invoke_memptr(f, std::forward<Args>(args)...);
    else
      return std::forward<F>(f)(std::forward<Args>(args)...);
  }
}
#endif


#if __cplusplus < 202302L
namespace OpenKalman
{
  /**
   * \internal
   * \brief A constexpr version of std::invoke_r, for use when compiling in c++17
   **/
#ifdef __cpp_concepts
  template<typename R, typename F, typename...Args> requires std::is_invocable_r_v<R, F, Args...>
#else
  template<typename R, typename F, typename...Args, std::enable_if_t<std::is_invocable_r_v<R, F, Args...>, int> = 0>
#endif
  constexpr R invoke_r(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable_r_v<R, F, Args...>)
  {
#if __cplusplus < 202002L
    namespace s = OpenKalman;
#else
    namespace s = std;
#endif
    if constexpr (std::is_void_v<R>)
      s::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    else
      return s::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }
}
#endif


namespace OpenKalman::internal
{
  namespace std_get_detail
  {
    using std::get;


#ifndef __cpp_concepts
    template<std::size_t i, typename T, typename = void>
    struct member_get_is_defined : std::false_type {};

    template<std::size_t i, typename T>
    struct member_get_is_defined<i, T, std::void_t<decltype(std::declval<T>().template get<i>())>> : std::true_type {};


    template<std::size_t i, typename T, typename = void>
    struct function_get_is_defined : std::false_type {};

    template<std::size_t i, typename T>
    struct function_get_is_defined<i, T, std::void_t<decltype(get<i>(std::declval<T>()))>> : std::true_type {};
#endif


    template<std::size_t i>
    struct get_impl
    {
#ifdef __cpp_concepts
      template<typename T> requires
        requires { std::declval<T&&>().template get<i>(); } or
        requires { get<i>(std::declval<T&&>()); }
#else
      template<typename T, std::enable_if_t<member_get_is_defined<i, T>::value or function_get_is_defined<i, T>::value, int> = 0>
#endif
      constexpr decltype(auto)
      operator() [[nodiscard]] (T&& t) const
      {
#ifdef __cpp_concepts
        if constexpr (requires { std::forward<T>(t).template get<i>(); })
#else
        if constexpr (member_get_is_defined<i, T>::value)
#endif
          return std::forward<T>(t).template get<i>();
        else
          return get<i>(std::forward<T>(t));
      }
    };
  }


  /**
   * \internal
   * \brief This is a placeholder for a more general <code>std::get</code> function that might be added to the standard library, possibly by another name.
   */
  template<std::size_t i>
  inline constexpr std_get_detail::get_impl<i>
  generalized_std_get;

}


#endif //OPENKALMAN_LANGUAGE_FEATURES_HPP
