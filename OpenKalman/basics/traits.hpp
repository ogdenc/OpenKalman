/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Declarations for OpenKalman and native-matrix traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  using namespace interface;

  // --------------------------------------- //
  //  constant-related classes and concepts  //
  // --------------------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    concept has_get_constant_interface = requires(interface::SingleConstant<std::decay_t<T>>& trait) {
      {trait.get_constant()} -> scalar_constant<c>;
    };
#else
    template<typename T, typename = void>
    struct get_constant_res {};

    template<typename T>
    struct get_constant_res<T, std::void_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant())>>
    {
      using type = std::decay_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant())>;
    };

    template<typename T, CompileTimeStatus c, typename = void>
    struct has_get_constant_interface_impl : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct has_get_constant_interface_impl<T, c, std::enable_if_t<scalar_constant<typename get_constant_res<T>::type, c>>>
      : std::true_type {};

    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    constexpr bool has_get_constant_interface = has_get_constant_interface_impl<T, c>::value;

    template<typename T, typename = void>
    struct constant_matrix_is_zero : std::false_type {};

    template<typename T>
    struct constant_matrix_is_zero<T, std::enable_if_t<are_within_tolerance(get_constant_res<T>::type::value, 0)>>
      : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    concept has_get_constant_diagonal_interface = requires(interface::SingleConstant<std::decay_t<T>>& trait) {
      {trait.get_constant_diagonal()} -> scalar_constant<c>;
    };
#else
    template<typename T, typename = void>
    struct get_constant_diagonal_res {};

    template<typename T>
    struct get_constant_diagonal_res<T, std::void_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant_diagonal())>>
    {
      using type = std::decay_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant_diagonal())>;
    };

    template<typename T, CompileTimeStatus c, typename = void>
    struct has_get_constant_diagonal_interface_impl : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct has_get_constant_diagonal_interface_impl<T, c, std::void_t<typename get_constant_diagonal_res<T>::type>>
      : std::bool_constant<scalar_constant<typename get_constant_diagonal_res<T>::type, c>> {};

    template<typename T, CompileTimeStatus c = CompileTimeStatus::any>
    constexpr bool has_get_constant_diagonal_interface = has_get_constant_diagonal_interface_impl<std::decay_t<T>, c>::value;

    template<typename T, typename = void>
    struct constant_diagonal_matrix_is_zero : std::false_type {};

    template<typename T>
    struct constant_diagonal_matrix_is_zero<T, std::void_t<decltype(get_constant_diagonal_res<T>::type::value)>>
      : std::bool_constant<are_within_tolerance(get_constant_diagonal_res<T>::type::value, 0)> {};
#endif


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct constant_status : std::integral_constant<Likelihood, Likelihood::definitely> {};

#ifdef __cpp_concepts
    template<typename T> requires (std::decay_t<T>::status == Likelihood::maybe)
    struct constant_status<T>
#else
    template<typename T>
    struct constant_status<T, std::enable_if_t<std::decay_t<T>::status == Likelihood::maybe>>
#endif
      : std::integral_constant<Likelihood, Likelihood::maybe> {};


    template<typename T>
    static constexpr bool is_1_by_1_impl(const T& t, std::index_sequence<>) { return true; }

    template<std::size_t I, std::size_t...Is, typename T>
    static constexpr bool is_1_by_1_impl(const T& t, std::index_sequence<I, Is...>)
    {
      if constexpr (dimension_size_of_index_is<T, I, 1>) return is_1_by_1_impl(t, std::index_sequence<Is...>{});
      else if (IndexTraits<T>::template dimension_at_runtime<I>(t) == 1) return is_1_by_1_impl(t, std::index_sequence<Is...>{});
      else return false;
    }

    template<typename T>
    static constexpr bool is_1_by_1(const T& t) { return is_1_by_1_impl(t, std::make_index_sequence<max_indices_of_v<T>>{}); }

    template<typename T, std::size_t...is>
    static constexpr auto get00element(const T& t, std::index_sequence<is...>)
    {
      return interface::Elements<std::decay_t<T>>::get(t, decltype(is){0}...);
    }
  } // namespace detail


  // -------------------------------------- //
  //  constant_coefficient specializations  //
  // -------------------------------------- //

  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::has_get_constant_interface<T, CompileTimeStatus::known> or
    (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> and
      (one_by_one_matrix<T, Likelihood::maybe> or requires(interface::SingleConstant<std::decay_t<T>>& trait) {
        requires are_within_tolerance(std::decay_t<decltype(trait.get_constant_diagonal())>::value, 0);
      }))
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<detail::has_get_constant_interface<T, CompileTimeStatus::known> or
    (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> and
      (one_by_one_matrix<T, Likelihood::maybe> or detail::constant_diagonal_matrix_is_zero<T>::value))>>
#endif
  {
  private:

    using Trait = interface::SingleConstant<std::decay_t<T>>;

  public:

    constexpr constant_coefficient() = default;

    explicit constexpr constant_coefficient(const std::decay_t<T>&) {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_coefficient;

    static constexpr value_type value = []{
      if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::known>)
        return std::decay_t<decltype(std::declval<Trait>().get_constant())>::value;
      else
        return std::decay_t<decltype(std::declval<Trait>().get_constant_diagonal())>::value;
    }();

    static constexpr Likelihood status = []{
      if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::known>)
        return detail::constant_status<decltype(std::declval<Trait>().get_constant())>::value;
      else
        return has_dynamic_dimensions<T> ? Likelihood::maybe : Likelihood::definitely;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }
  };


  /**
   * \internal
   * \brief Specialization of \ref constant_coefficient in which the constant can be known only at runtime.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::has_get_constant_interface<T, CompileTimeStatus::unknown> or
    (one_by_one_matrix<T, Likelihood::maybe> and (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown> or
      (not detail::has_get_constant_interface<T> and not detail::has_get_constant_diagonal_interface<T>)))
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<detail::has_get_constant_interface<T, CompileTimeStatus::unknown> or
    (one_by_one_matrix<T, Likelihood::maybe> and (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown> or
      (not detail::has_get_constant_interface<T> and not detail::has_get_constant_diagonal_interface<T>)))>>
#endif
  {
  private:

    using Trait = interface::SingleConstant<std::decay_t<T>>;

  public:

    explicit constexpr constant_coefficient(const std::decay_t<T>& t) : value {[](const auto& t){
        if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::unknown>)
        {
          return get_scalar_constant_value(Trait{t}.get_constant());
        }
        else
        {
          if constexpr (not one_by_one_matrix<T>) if (not detail::is_1_by_1(t)) throw std::logic_error {"Not a constant object"};
          if constexpr (detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown>)
            return get_scalar_constant_value(Trait{t}.get_constant_diagonal());
          else
            return detail::get00element(t, std::make_index_sequence<max_indices_of_v<T>>{});
        }
      }(t)} {};

    using value_type = scalar_type_of_t<T>;

    using type = constant_coefficient;

    static constexpr Likelihood status = []{
      if constexpr (detail::has_get_constant_interface<T, CompileTimeStatus::unknown>)
        return detail::constant_status<decltype(std::declval<Trait>().get_constant())>::value;
      else
        return has_dynamic_dimensions<T> ? Likelihood::maybe : Likelihood::definitely;
    }();

    constexpr operator value_type() const noexcept { return value; }

    constexpr value_type operator()() const noexcept { return value; }

  private:
    value_type value;
  };


  // ----------------------------------------------- //
  //  constant_diagonal_coefficient specializations  //
  // ----------------------------------------------- //

  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant is known at compile time.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> or
    (detail::has_get_constant_interface<T, CompileTimeStatus::known> and
      (one_by_one_matrix<T, Likelihood::maybe> or (square_matrix<T, Likelihood::maybe> and
        requires(interface::SingleConstant<std::decay_t<T>>& trait) {
          requires are_within_tolerance(std::decay_t<decltype(trait.get_constant())>::value, 0);
        })))
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::known> or
    (detail::has_get_constant_interface<T, CompileTimeStatus::known> and
      (one_by_one_matrix<T, Likelihood::maybe> or (square_matrix<T, Likelihood::maybe> and
      detail::constant_matrix_is_zero<T>::value)))>>
#endif
  {
  private:
    using Trait = interface::SingleConstant<std::decay_t<T>>;

  public:
    constexpr constant_diagonal_coefficient() = default;
    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>&) {};
    using value_type = scalar_type_of_t<T>;
    using type = constant_diagonal_coefficient;
    static constexpr value_type value = []{
      if constexpr (detail::has_get_constant_diagonal_interface<T>)
        return std::decay_t<decltype(std::declval<Trait>().get_constant_diagonal())>::value;
      else
        return std::decay_t<decltype(std::declval<Trait>().get_constant())>::value;
    }();
    static constexpr Likelihood status = []{
      if constexpr (has_dynamic_dimensions<T>)
        return Likelihood::maybe;
      else if constexpr (detail::has_get_constant_diagonal_interface<T>)
        return detail::constant_status<decltype(std::declval<Trait>().get_constant_diagonal())>::value;
      else
        return Likelihood::definitely;
    }();
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
  };


  /**
   * \internal
   * \brief Specialization of \ref constant_diagonal_coefficient in which the constant can be known only at runtime.
   */
#ifdef __cpp_concepts
  template<indexible T> requires detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown> or
    (one_by_one_matrix<T, Likelihood::maybe> and (detail::has_get_constant_interface<T, CompileTimeStatus::unknown> or
      (not detail::has_get_constant_diagonal_interface<T> and not detail::has_get_constant_interface<T>)))
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<detail::has_get_constant_diagonal_interface<T, CompileTimeStatus::unknown> or
    (one_by_one_matrix<T, Likelihood::maybe> and (detail::has_get_constant_interface<T, CompileTimeStatus::unknown> or
      (not detail::has_get_constant_diagonal_interface<T> and not detail::has_get_constant_interface<T>)))>>
#endif
  {
  private:
    using Trait = interface::SingleConstant<std::decay_t<T>>;

  public:
    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>& t) : value {[](const auto& t){
        if constexpr (detail::has_get_constant_diagonal_interface<T>)
        {
          return get_scalar_constant_value(Trait{t}.get_constant_diagonal());
        }
        else
        {
          if constexpr (not one_by_one_matrix<T>) if (not detail::is_1_by_1(t)) throw std::logic_error {"Not a diagonal constant object"};
          if constexpr (detail::has_get_constant_interface<T>)
            return get_scalar_constant_value(Trait{t}.get_constant());
          else
            return detail::get00element(t, std::make_index_sequence<max_indices_of_v<T>>{});
        }
      }(t)} {};

    using value_type = scalar_type_of_t<T>;
    using type = constant_diagonal_coefficient;
    static constexpr Likelihood status = []{
      if constexpr (has_dynamic_dimensions<T>)
        return Likelihood::maybe;
      else if constexpr (detail::has_get_constant_diagonal_interface<T>)
        return detail::constant_status<decltype(std::declval<Trait>().get_constant_diagonal())>::value;
      else
        return Likelihood::definitely;
    }();
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }

  private:
    value_type value;
  };


  // --------------- //
  //  typed_adapter  //
  // --------------- //

  /**
   * \brief Specifies that T is a typed adapter expression.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_adapter =
#else
  constexpr bool typed_adapter =
#endif
    typed_matrix<T> or covariance<T> or euclidean_expr<T>;


  // ----------------- //
  //  untyped_adapter  //
  // ----------------- //

  /**
   * \brief Specifies that T is an untyped adapter expression.
   * \details Untyped adapter expressions are generally used whenever the native matrix library does not have an
   * important built-in matrix type, such as a diagonal matrix, a triangular matrix, or a hermitian matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept untyped_adapter =
#else
  constexpr bool untyped_adapter =
#endif
    eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>;


  // ========= //
  //  Aliases  //
  // ========= //

  // ------------------------- //
  //  dense_writable_matrix_t  //
  // ------------------------- //

  namespace detail
  {
    template<typename T, typename Scalar, typename...D>
    struct dense_writable_matrix_impl
    {
      using type = std::decay_t<decltype(make_default_dense_writable_matrix_like<T, Scalar>(std::declval<D>()...))>;
    };


    template<typename T, typename Scalar>
    struct dense_writable_matrix_impl<T, Scalar>
    {
      using type = std::decay_t<decltype(make_default_dense_writable_matrix_like<Scalar>(std::declval<T>()))>;
    };
  }


  /**
    * \brief An alias for a dense, writable matrix, patterned on parameter T.
    * \tparam T A matrix or array from the relevant matrix library.
    * \tparam S A scalar type (may or may not be </code>scalar_type_of_t<T></code>.
    * \tparam D Index descriptors defining the dimensions of the new matrix.
    * \todo Create typed Matrix if Ds are typed.
    */
#ifdef __cpp_concepts
  template<indexible T, scalar_type S = scalar_type_of_t<T>, index_descriptor...D>
#else
  template<typename T, typename S = scalar_type_of_t<T>, typename...D>
#endif
  using dense_writable_matrix_t = typename detail::dense_writable_matrix_impl<T, std::decay_t<S>, D...>::type;


  // --------------------------------- //
  //  untyped_dense_writable_matrix_t  //
  // --------------------------------- //

  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam D Integral values defining the dimensions of the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type S, auto...D> requires ((std::is_integral_v<decltype(D)> and D >= 0) and ...)
#else
  template<typename T, typename S, auto...D>
#endif
  using untyped_dense_writable_matrix_t = dense_writable_matrix_t<T, S, Dimensions<static_cast<const std::size_t>(D)>...>;


  // --------------------------- //
  //  equivalent_self_contained  //
  // --------------------------- //

  /**
   * \brief An alias for type, derived from and equivalent to parameter T, that is self-contained.
   * \details Use this alias to obtain a type, equivalent to T, that can safely be returned from a function.
   * \sa self_contained, make_self_contained
   * \internal \sa interface::Dependencies
   */
  template<typename T>
  using equivalent_self_contained_t = std::remove_reference_t<decltype(make_self_contained(std::declval<T>()))>;


  // ------------ //
  //  passable_t  //
  // ------------ //

  /**
   * \brief An alias for a type, derived from and equivalent to parameter T, that can be passed as a function parameter.
   * \tparam T The type in question.
   * \details A passable type T is either an lvalue reference or is \ref equivalent_self_contained_t.
   */
  template<typename T>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, equivalent_self_contained_t<T>>;


  namespace internal
  {
    // --------------- //
    //  is_modifiable  //
    // --------------- //

#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct has_const : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires std::is_const_v<std::remove_reference_t<T>> or
      (requires { typename nested_matrix_of_t<T>; } and has_const<nested_matrix_of_t<T>>::value)
    struct has_const<T> : std::true_type {};
#else
    template<typename T>
    struct has_const<T, std::enable_if_t<std::is_const_v<std::remove_reference_t<T>>>> : std::true_type {};

    template<typename T>
    struct has_const<T, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>) and
      has_const<nested_matrix_of_t<T>>::value>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      has_const<T>::value or
      (not maybe_has_same_shape_as<T, U>) or
      (not std::same_as<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (triangular_matrix<T, TriangleType::upper> and not triangular_matrix<U, TriangleType::upper>) or
      (triangular_matrix<T, TriangleType::lower> and not triangular_matrix<U, TriangleType::lower>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)
    struct is_modifiable<T, U> : std::false_type {};
#else
    template<typename T, typename U>
    struct is_modifiable<T, U, std::enable_if_t<
      has_const<T>::value or
      (not maybe_has_same_shape_as<T, U>) or
      (not std::is_same_v<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (triangular_matrix<T, TriangleType::upper> and not triangular_matrix<U, TriangleType::upper>) or
      (triangular_matrix<T, TriangleType::lower> and not triangular_matrix<U, TriangleType::lower>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)>> : std::false_type {};
#endif

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_TRAITS_HPP
