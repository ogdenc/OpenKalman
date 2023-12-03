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
 * \brief SelfAdjointMatrix and related definitions.
 */

#ifndef OPENKALMAN_SELFADJOINTMATRIX_HPP
#define OPENKALMAN_SELFADJOINTMATRIX_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> NestedMatrix, HermitianAdapterType storage_triangle> requires
    (index_count_v<NestedMatrix> <= 2) and
    (storage_triangle == HermitianAdapterType::lower or storage_triangle == HermitianAdapterType::upper) and
    (not constant_matrix<NestedMatrix> or real_axis_number<constant_coefficient<NestedMatrix>>) and
    (not constant_diagonal_matrix<NestedMatrix> or real_axis_number<constant_diagonal_coefficient<NestedMatrix>>) and
    (not triangular_matrix<NestedMatrix, TriangleType::any, Likelihood::maybe> or
      triangular_matrix<NestedMatrix, static_cast<TriangleType>(storage_triangle), Likelihood::maybe>)
#else
  template<typename NestedMatrix, HermitianAdapterType storage_triangle>
#endif
  struct SelfAdjointMatrix
    : OpenKalman::internal::MatrixBase<SelfAdjointMatrix<NestedMatrix, storage_triangle>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(square_matrix<NestedMatrix, Likelihood::maybe>);
    static_assert(index_count_v<NestedMatrix> <= 2);
    static_assert(storage_triangle == HermitianAdapterType::lower or storage_triangle == HermitianAdapterType::upper);
    static_assert([]{if constexpr (constant_matrix<NestedMatrix>) return real_axis_number<constant_coefficient<NestedMatrix>>; else return true; }());
    static_assert([]{if constexpr (constant_diagonal_matrix<NestedMatrix>) return real_axis_number<constant_diagonal_coefficient<NestedMatrix>>; else return true; }());
    static_assert(not triangular_matrix<NestedMatrix, TriangleType::any, Likelihood::maybe> or
      triangular_matrix<NestedMatrix, static_cast<TriangleType>(storage_triangle), Likelihood::maybe>);
#endif


  private:

    using Base = OpenKalman::internal::MatrixBase<SelfAdjointMatrix, NestedMatrix>;

    static constexpr auto dim = dynamic_dimension<NestedMatrix, 0> ? index_dimension_of_v<NestedMatrix, 1> :
      index_dimension_of_v<NestedMatrix, 0>;


  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


    /// Default constructor.
#ifdef __cpp_concepts
    SelfAdjointMatrix() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    SelfAdjointMatrix()
#endif
      : Base {} {}


    /// Copy constructor.
    SelfAdjointMatrix(const SelfAdjointMatrix& other) : Base {other} {}


    /// Move constructor.
    SelfAdjointMatrix(SelfAdjointMatrix&& other) noexcept : Base {std::move(other)} {}


    /// Construct from a diagonal matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      diagonal_matrix<NestedMatrix> and has_same_shape_as<Arg, NestedMatrix> and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      diagonal_matrix<Arg> and diagonal_matrix<NestedMatrix> and has_same_shape_as<Arg, NestedMatrix> and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a diagonal matrix if NestedMatrix is not diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (not diagonal_matrix<NestedMatrix> or
        not requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      (not diagonal_matrix<NestedMatrix> or
        not std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian, non-diagonal wrapper of the same storage type
#ifdef __cpp_concepts
    template<hermitian_adapter<storage_triangle> Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (not diagonal_matrix<Arg>) and square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<
      hermitian_adapter<Arg, storage_triangle> and (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      (not diagonal_matrix<Arg>) and square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a hermitian, non-diagonal wrapper of the opposite storage type
#ifdef __cpp_concepts
    template<hermitian_adapter Arg> requires (not diagonal_matrix<Arg>) and
      (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      requires(Arg&& arg) { NestedMatrix {transpose(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<
      hermitian_adapter<Arg> and (not diagonal_matrix<Arg>) and
      (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      std::is_constructible_v<NestedMatrix, decltype(transpose(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {transpose(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from a hermitian matrix of the same storage type and is not a wrapper
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value == storage_triangle) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value == storage_triangle) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian matrix, of the opposite storage type, that is not a wrapper
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      requires(Arg&& arg) { NestedMatrix {transpose(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      std::is_constructible_v<NestedMatrix, decltype(transpose(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {transpose(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedMatrix is not diagonal.
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not hermitian_adapter<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
      not hermitian_adapter<Arg> and not eigen_diagonal_expr<NestedMatrix> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept : Base {
      [](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_dimension<Arg, 0> and dim != dynamic_size) assert(get_index_dimension_of<0>(arg) == dim);
        if constexpr (dynamic_dimension<Arg, 1> and dim != dynamic_size) assert(get_index_dimension_of<1>(arg) == dim);
        return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not hermitian_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and (not hermitian_matrix<Arg>) and
      diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept : Base {
      [](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_dimension<Arg, 0> and dim != dynamic_size) assert(get_index_dimension_of<0>(arg) == dim);
        if constexpr (dynamic_dimension<Arg, 1> and dim != dynamic_size) assert(get_index_dimension_of<1>(arg) == dim);
        return diagonal_of(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { NestedMatrix {make_dense_writable_matrix_from<NestedMatrix>(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<
      std::conjunction_v<std::is_convertible<Args, const Scalar>...> and (sizeof...(Args) > 0) and
      (std::is_constructible_v<NestedMatrix,
        untyped_dense_writable_matrix_t<NestedMatrix, Layout::none, Scalar, static_cast<std::size_t>(
          internal::constexpr_sqrt(sizeof...(Args))), static_cast<std::size_t>(internal::constexpr_sqrt(sizeof...(Args)))>> or
        (diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix,
          untyped_dense_writable_matrix_t<NestedMatrix, Layout::none, Scalar, sizeof...(Args), 1>>)), int> = 0>
#endif
    SelfAdjointMatrix(Args ... args)
      : Base {make_dense_writable_matrix_from<NestedMatrix>(static_cast<const Scalar>(args)...)} {}


    /** Copy assignment operator
     * \param other Another SelfAdjointMatrix
     * \return Reference to this.
     */
    auto& operator=(const SelfAdjointMatrix& other)
    {
      if constexpr (not constant_matrix<NestedMatrix, CompileTimeStatus::known> and
          not constant_diagonal_matrix<NestedMatrix, CompileTimeStatus::known>);
      else if (this != &other)
      {
        if constexpr (writable<NestedMatrix>) internal::set_triangle<storage_triangle>(this->nested_matrix(), other.nested_matrix());
        else Base::operator=(other);
      }

      return *this;
    }


    /** Move assignment operator
     * \param other A SelfAdjointMatrix temporary value.
     * \return Reference to this.
     */
    auto& operator=(SelfAdjointMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another \ref hermitian_matrix.
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      maybe_has_same_shape_as<NestedMatrix, Arg> and (not diagonal_matrix<NestedMatrix> or diagonal_matrix<Arg>) and
      (writable<NestedMatrix> or std::assignable_from<NestedMatrix, Arg&&>)
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and
      (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and maybe_has_same_shape_as<NestedMatrix, Arg> and
      (not diagonal_matrix<NestedMatrix> or diagonal_matrix<Arg>) and
      (writable<NestedMatrix> or std::is_assignable_v<NestedMatrix, Arg&&>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (writable<NestedMatrix>) internal::set_triangle<storage_triangle>(this->nested_matrix(), std::forward<Arg>(arg));
      else if (hermitian_adapter_type_of_v<Arg> != storage_triangle) Base::operator=(adjoint(std::forward<Arg>(arg)));
      else Base::operator=(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, HermitianAdapterType t> requires diagonal_matrix<Arg> or
      (not diagonal_matrix<NestedMatrix>)
#else
    template<typename Arg, HermitianAdapterType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix> and
      (diagonal_matrix<Arg> or (not diagonal_matrix<NestedMatrix>)), int> = 0>
#endif
    auto& operator+=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      if constexpr (writable<NestedMatrix>) internal::set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() + std::forward<Arg>(arg));
      else if (t != storage_triangle) Base::operator+=(adjoint(std::forward<Arg>(arg)));
      else Base::operator+=(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, HermitianAdapterType t> requires diagonal_matrix<Arg> or
      (not diagonal_matrix<NestedMatrix>)
#else
    template<typename Arg, HermitianAdapterType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix> and
      (diagonal_matrix<Arg> or (not diagonal_matrix<NestedMatrix>)), int> = 0>
#endif
    auto& operator-=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      if constexpr (writable<NestedMatrix>) internal::set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() - std::forward<Arg>(arg));
      else if (t != storage_triangle) Base::operator-=(adjoint(std::forward<Arg>(arg)));
      else Base::operator-=(std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr (writable<NestedMatrix>) internal::set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() * s);
      else Base::operator*=(s);
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      if constexpr (writable<NestedMatrix>) internal::set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() / s);
      else Base::operator/=(s);
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, SelfAdjointMatrix>
    friend decltype(auto) operator-(Arg&& arg)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(-nested_matrix(std::forward<Arg>(arg)));
    }
#else
    decltype(auto) operator-() const&
    {
      return make_hermitian_matrix<storage_triangle>(-nested_matrix(*this));
    }

    decltype(auto) operator-() const&&
    {
      return make_hermitian_matrix<storage_triangle>(-nested_matrix(std::move(*this)));
    }
#endif


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, SelfAdjointMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, SelfAdjointMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(Arg&& arg, S s)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(nested_matrix(std::forward<Arg>(arg)) * s);
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, SelfAdjointMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, SelfAdjointMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator*(S s, Arg&& arg)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(s * nested_matrix(std::forward<Arg>(arg)));
    }


#ifdef __cpp_concepts
    template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>> S> requires std::same_as<std::decay_t<Arg>, SelfAdjointMatrix>
#else
    template<typename Arg, typename S, std::enable_if_t<
      std::is_same_v<std::decay_t<Arg>, SelfAdjointMatrix> and std::is_convertible_v<S, const scalar_type_of_t<Arg>>>>
#endif
    friend decltype(auto) operator/(Arg&& arg, S s)
    {
      return make_hermitian_matrix<hermitian_adapter_type_of_v<Arg>>(nested_matrix(std::forward<Arg>(arg)) / s);
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<hermitian_matrix<Likelihood::maybe> M>
#else
  template<typename M, std::enable_if_t<hermitian_matrix<M, Likelihood::maybe>, int> = 0>
#endif
  SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<
    std::conditional_t<hermitian_adapter<M>, passable_t<nested_matrix_of_t<M&&>>, passable_t<M>>,
    hermitian_adapter<M> ? hermitian_adapter_type_of_v<M> : HermitianAdapterType::lower>;


#ifdef __cpp_concepts
  template<triangular_matrix<TriangleType::any, Likelihood::maybe> M> requires (not hermitian_matrix<M, Likelihood::maybe>)
#else
  template<typename M, std::enable_if_t<triangular_matrix<M, TriangleType::any, Likelihood::maybe> and
    (not hermitian_matrix<M, Likelihood::maybe>), int> = 0>
#endif
  explicit SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<
    std::conditional_t<triangular_adapter<M>, passable_t<nested_matrix_of_t<M&&>>, passable_t<M>>,
    triangular_matrix<M, TriangleType::lower> ? HermitianAdapterType::lower : HermitianAdapterType::upper>;


#ifdef __cpp_concepts
  template<indexible M> requires (not hermitian_matrix<M, Likelihood::maybe>) and
    (not triangular_matrix<M, TriangleType::any, Likelihood::maybe>)
#else
  template<typename M, std::enable_if_t<indexible<M> and (not hermitian_matrix<M, Likelihood::maybe>) and
      (not triangular_matrix<M, TriangleType::any, Likelihood::maybe>), int> = 0>
#endif
  explicit SelfAdjointMatrix(M&&) -> SelfAdjointMatrix<passable_t<M>, HermitianAdapterType::lower>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename NestedMatrix, HermitianAdapterType storage_type>
    struct indexible_object_traits<SelfAdjointMatrix<NestedMatrix, storage_type>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg>
      static constexpr auto get_index_count(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N n)
      {
        if constexpr (static_index_value<N>)
        {
          if constexpr (dynamic_dimension<NestedMatrix, 0>) return OpenKalman::get_vector_space_descriptor<1>(nested_matrix(std::forward<Arg>(arg)));
          else return OpenKalman::get_vector_space_descriptor<0>(nested_matrix(std::forward<Arg>(arg)));
        }
        else
        {
          return OpenKalman::get_vector_space_descriptor<0>(nested_matrix(arg));
        }
      }

      using dependents = std::tuple<NestedMatrix>;

      static constexpr bool has_runtime_parameters = false;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(get_nested_matrix(std::forward<Arg>(arg)));
        return SelfAdjointMatrix<decltype(n), storage_type> {std::move(n)};
      }

      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        return constant_coefficient{nested_matrix(arg)};
      }

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        return constant_diagonal_coefficient {nested_matrix(arg)};
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<NestedMatrix, b>;

      template<TriangleType t, Likelihood>
      static constexpr bool is_triangular = triangular_matrix<NestedMatrix, TriangleType::diagonal, Likelihood::maybe>;

      static constexpr bool is_hermitian = true;

      static constexpr HermitianAdapterType hermitian_adapter_type = storage_type;


#ifdef __cpp_lib_concepts
    template<diagonal_matrix Arg, typename I> requires element_gettable<nested_matrix_of_t<Arg&&>, 1> or
      element_gettable<nested_matrix_of_t<Arg&&>, 2>
#else
    template<typename Arg, typename I, std::enable_if_t<diagonal_matrix<Arg> and
      element_gettable<typename nested_matrix_of<Arg&&>::type, 1> and
      element_gettable<typename nested_matrix_of<Arg&&>::type, 2>, int> = 0>
#endif
      static constexpr auto get(Arg&& arg, I i)
      {
        if constexpr (element_gettable<nested_matrix_of_t<Arg&&>, 1>)
          return get_element(nested_matrix(std::forward<Arg>(arg)), i);
        else
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, i);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename J> requires element_gettable<nested_matrix_of_t<Arg&&>, 2>
  #else
      template<typename Arg, typename I, typename J, std::enable_if_t<
        element_gettable<typename nested_matrix_of<Arg&&>::type, 2>, int> = 0>
  #endif
      static constexpr scalar_type_of_t<Arg> get(Arg&& arg, I i, J j)
      {
        using Scalar = scalar_type_of<Arg>;

        auto&& n = nested_matrix(std::forward<Arg>(arg));
        using N = decltype(n);

        if (hermitian_adapter<Arg, HermitianAdapterType::lower> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
        {
          if constexpr (complex_number<Scalar>)
          {
            if (i == j) return internal::constexpr_real(get_element(std::forward<N>(n), i, j));
          }
          return get_element(std::forward<decltype(n)>(n), i, j);
        }
        else
        {
          if constexpr (complex_number<Scalar>)
            return internal::constexpr_conj(get_element(std::forward<N>(n), j, i));
          else
            return get_element(std::forward<N>(n), j, i);
        }
      }


#ifdef __cpp_lib_concepts
    template<diagonal_matrix Arg, typename I> requires element_settable<nested_matrix_of_t<Arg&>, 1> or
      element_settable<nested_matrix_of_t<Arg&>, 2>
#else
    template<typename Arg, typename I, std::enable_if_t<diagonal_matrix<Arg> and
      element_settable<typename nested_matrix_of<Arg&>::type, 1> and
      element_settable<typename nested_matrix_of<Arg&>::type, 2>, int> = 0>
#endif
      static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i)
      {
        if constexpr (element_settable<nested_matrix_of_t<Arg&>, 1>)
          set_element(nested_matrix(arg), s, i);
        else
          set_element(nested_matrix(arg), s, i, static_cast<I>(1));
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename J> requires element_settable<nested_matrix_of_t<Arg&>, 2>
  #else
      template<typename Arg, typename I, typename J, std::enable_if_t<element_settable<typename nested_matrix_of<Arg&>::type, 2>, int> = 0>
  #endif
      static void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, J j)
      {
        if (hermitian_adapter<Arg, HermitianAdapterType::lower> ? i >= static_cast<I>(j) : i <= static_cast<I>(j))
          set_element(nested_matrix(arg), s, i, j);
        else
          set_element(nested_matrix(arg), internal::constexpr_conj(s), j, i);
      }


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires directly_accessible<nested_matrix_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<directly_accessible<typename nested_matrix_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto*
      data(Arg& arg) { return internal::raw_data(nested_matrix(arg)); }


      static constexpr Layout layout = one_by_one_matrix<NestedMatrix> ? layout_of_v<NestedMatrix> : Layout::none;

    };

  } // namespace interface

} // namespace OpenKalman



#endif //OPENKALMAN_SELFADJOINTMATRIX_HPP

