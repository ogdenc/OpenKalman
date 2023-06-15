/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for Eigen3::EigenWrapper
 */

#ifndef OPENKALMAN_EIGENWRAPPER_HPP
#define OPENKALMAN_EIGENWRAPPER_HPP

namespace OpenKalman
{
  namespace Eigen3
  {
    template<typename NestedMatrix>
    struct EigenWrapper : std::conditional_t<
      std::is_base_of_v<Eigen::ArrayBase<NestedMatrix>, std::decay_t<NestedMatrix>>,
      Eigen::ArrayBase<EigenWrapper<NestedMatrix>>,
      Eigen::MatrixBase<EigenWrapper<NestedMatrix>>>
    {
    private:

      using Base = std::conditional_t<
        std::is_base_of_v<Eigen::ArrayBase<NestedMatrix>, std::decay_t<NestedMatrix>>,
        Eigen::ArrayBase<EigenWrapper>,
        Eigen::MatrixBase<EigenWrapper>>;


#ifdef __cpp_concepts
      template<typename T>
#else
      template<typename T, typename = void>
#endif
      struct PacketScalarImpl { using type = scalar_type_of<NestedMatrix>; };


#ifdef __cpp_concepts
      template<typename T> requires requires { typename T::PacketScalar; }
      struct PacketScalarImpl<T>
#else
      template<typename T>
      struct PacketScalarImpl<T, std::void_t<typename T::PacketScalar>>
#endif
        { using type = typename T::PacketScalar; };


    public:

#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, EigenWrapper>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
      template<typename Arg, std::enable_if_t<(not std::is_same_v<std::decay_t<Arg>, EigenWrapper>) and
        std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
      explicit EigenWrapper(Arg&& arg) : wrapped_expression {std::forward<Arg>(arg)} {}


#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, EigenWrapper>) and
      (not std::constructible_from<NestedMatrix, Arg&&>) and std::default_initializable<NestedMatrix> and writable<NestedMatrix>
#else
    template<typename Arg, std::enable_if_t<(not std::is_same_v<std::decay_t<Arg>, EigenWrapper>) and
      (not std::is_constructible_v<NestedMatrix, Arg&&>) and std::is_default_constructible_v<NestedMatrix> and writable<NestedMatrix>, int> = 0>
#endif
      explicit EigenWrapper(Arg&& arg)
      {
        for (std::size_t i = 0; i < get_index_dimension_of<0>(wrapped_expression); i++)
        for (std::size_t j = 0; j < get_index_dimension_of<1>(wrapped_expression); j++)
          set_element(wrapped_expression, get_element(std::forward<Arg>(arg), i, j), i, j);
      }


      /**
       * \brief Get the nested matrix.
       */
      decltype(auto) nested_matrix() & noexcept { return (wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() const & noexcept { return (wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() && noexcept { return (std::move(*this).wrapped_expression); }

      /// \overload
      decltype(auto) nested_matrix() const && noexcept { return (std::move(*this).wrapped_expression); }


      /// \internal \note Eigen3 requires this.
      using Scalar = scalar_type_of_t<NestedMatrix>;

      using PacketScalar = typename PacketScalarImpl<Base>::type;


      /* \internal
       * \brief The underlying numeric type for composed scalar types.
       * \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar.
       */
      using RealScalar = typename Eigen::NumTraits<Scalar>::Real;


      /* \internal
       * \brief The return type for coefficient access.
       * \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is
       * either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access.
       */
      //using CoeffReturnType = typename Base::CoeffReturnType;
      using CoeffReturnType = std::conditional_t<bool(Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags & Eigen::LvalueBit),
        const Scalar&, std::conditional_t<std::is_arithmetic_v<Scalar>, Scalar, const Scalar>>;


      /**
       * \internal
       * \brief The type of *this that is used for nesting within other Eigen classes.
       * \note Eigen3 requires this as the type used when Derived is nested.
       */
      using Nested = EigenWrapper;

      using NestedExpression = std::decay_t<NestedMatrix>;

      using StorageKind [[maybe_unused]] = typename Eigen::internal::traits<EigenWrapper>::StorageKind;

      using StorageIndex [[maybe_unused]] = typename Eigen::internal::traits<EigenWrapper>::StorageIndex;


      enum CompileTimeTraits
      {
        RowsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<EigenWrapper>::RowsAtCompileTime,
        ColsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<EigenWrapper>::ColsAtCompileTime,
        MaxRowsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<EigenWrapper>::MaxRowsAtCompileTime,
        MaxColsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<EigenWrapper>::MaxColsAtCompileTime,
        Flags [[maybe_unused]] = Eigen::internal::traits<EigenWrapper>::Flags,
        SizeAtCompileTime [[maybe_unused]] = (Eigen::internal::size_at_compile_time<RowsAtCompileTime, ColsAtCompileTime>::ret),
        MaxSizeAtCompileTime [[maybe_unused]] = (Eigen::internal::size_at_compile_time<MaxRowsAtCompileTime, MaxColsAtCompileTime>::ret),
        IsVectorAtCompileTime [[maybe_unused]] = RowsAtCompileTime == 1 or ColsAtCompileTime == 1,
      };


      /**
       * \internal
       * \return The number of rows at runtime.
       * \note Eigen3 requires this, particularly in Eigen::EigenBase.
       */
      constexpr Eigen::Index rows() const
      {
        return get_index_dimension_of<0>(wrapped_expression);
      }


      /**
       * \internal
       * \return The number of columns at runtime.
       * \note Eigen3 requires this, particularly in Eigen::EigenBase.
       */
      constexpr Eigen::Index cols() const
      {
        return get_index_dimension_of<1>(wrapped_expression);
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) outerStride() const requires native_eigen_dense<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<native_eigen_dense<T>, int> = 0>
      constexpr decltype(auto) outerStride() const
#endif
      {
        return wrapped_expression.outerStride();
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) innerStride() const requires native_eigen_dense<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<native_eigen_dense<T>, int> = 0>
      constexpr decltype(auto) innerStride() const
#endif
      {
        return wrapped_expression.innerStride();
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) data() requires native_eigen_plain_object<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<native_eigen_plain_object<T>, int> = 0>
      constexpr decltype(auto) data()
#endif
      {
        return wrapped_expression.data();
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) data() const requires native_eigen_plain_object<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<native_eigen_plain_object<T>, int> = 0>
      constexpr decltype(auto) data() const
#endif
      {
        return wrapped_expression.data();
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) resize(Eigen::Index newSize) requires native_eigen_dense<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<native_eigen_dense<T>, int> = 0>
      constexpr decltype(auto) resize(Eigen::Index newSize)
#endif
      {
        return wrapped_expression.resize(newSize);
      }


#ifdef __cpp_concepts
      constexpr decltype(auto) resize(Eigen::Index rows, Eigen::Index cols) requires native_eigen_dense<NestedMatrix>
#else
      template<typename T = NestedMatrix, std::enable_if_t<native_eigen_dense<T>, int> = 0>
      constexpr decltype(auto) resize(Eigen::Index rows, Eigen::Index cols)
#endif
      {
        return wrapped_expression.resize(rows, cols);
      }

    private:

      using ElementRef = decltype(OpenKalman::get_element(std::declval<NestedMatrix>(), 0, 0));
      static constexpr bool lvalue_get_element = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>;

    public:

#ifdef __cpp_concepts
      auto& coeffRef(Eigen::Index row, Eigen::Index col) requires lvalue_get_element
#else
      template<bool b = lvalue_get_element, std::enable_if_t<b, int> = 0>
      auto& coeffRef(Eigen::Index row, Eigen::Index col)
#endif
      {
        return get_element(wrapped_expression, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


      constexpr decltype(auto) coeff(Eigen::Index row, Eigen::Index col) const
      {
        return get_element(wrapped_expression, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


      /**
       * \brief Synonym for zero().
       * \note Overrides Eigen::DenseBase<Derived>::Zero.
       * \return A matrix, of the same size and shape, containing only zero coefficients.
       */
      [[deprecated("Use make_zero_matrix_like() instead.")]]
      constexpr auto Zero()
      {
        static_assert(not has_dynamic_dimensions<NestedMatrix>);
        return make_zero_matrix_like(wrapped_expression);
      }


      /**
       * \brief Synonym for zero().
       * \note Overrides Eigen::DenseBase<Derived>::Zero.
       * \return A matrix, of the same size and shape, containing only zero coefficients.
       */
      [[deprecated("Use make_zero_matrix_like() instead.")]]
      static constexpr auto Zero(const Eigen::Index r, const Eigen::Index c)
      {
        return make_zero_matrix_like<NestedMatrix>(Dimensions{static_cast<std::size_t>(r)}, Dimensions{static_cast<std::size_t>(c)});
      }


      /**
       * \brief Synonym for identity().
       * \note Overrides Eigen::DenseBase<Derived>::Identity.
       * \return An identity matrix with the same or identified number of rows and columns.
       */
      [[deprecated("Use make_identity_matrix_like() instead.")]]
      constexpr auto Identity()
      {
        if constexpr(square_matrix<NestedMatrix>)
          return make_identity_matrix_like(wrapped_expression);
        else
          return std::decay_t<NestedMatrix>::Identity();
      }


#ifdef __cpp_concepts
      template<std::convertible_to<Scalar> S>
#else
      template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
      auto operator<<(const S& s)
      {
        return wrapped_expression.operator<<(s);
      }


#ifdef __cpp_concepts
      template<indexible Other>
#else
      template<typename Other, std::enable_if_t<indexible<Other>, int> = 0>
#endif
      auto operator<<(const Other& other)
      {
        return wrapped_expression.operator<<(other);
      }

    protected:

      NestedMatrix wrapped_expression;

    };


    /**
     * \brief Deduction guide for EigenWrapper
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not eigen_wrapper<Arg>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not eigen_wrapper<Arg>, int> = 0>
#endif
    EigenWrapper(Arg&&) -> EigenWrapper<Arg>;


  } // namespace Eigen3


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename NestedMatrix>
    struct IndexTraits<Eigen3::EigenWrapper<NestedMatrix>>
    {
      static constexpr std::size_t max_indices = max_indices_of_v<NestedMatrix>;

      template<std::size_t N, typename Arg>
      static constexpr auto get_index_descriptor(const Arg& arg)
      {
        return OpenKalman::get_index_descriptor<N>(nested_matrix(arg));
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<NestedMatrix, b>;

      template<Likelihood b>
      static constexpr bool is_square = square_matrix<NestedMatrix, b>;
    };


    template<typename T>
    struct Elements<Eigen3::EigenWrapper<T>>
    {
      using scalar_type = scalar_type_of_t<T>;

#ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires element_gettable<nested_matrix_of_t<Arg>, sizeof...(I)>
#else
      template<typename Arg, typename...I, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg>::type, sizeof...(I)>, int> = 0>
#endif
      static constexpr decltype(auto) get(Arg&& arg, I...i)
      {
        return get_element(nested_matrix(std::forward<Arg>(arg)), i...);
      }


#ifdef __cpp_lib_concepts
      template<typename Arg, typename Scalar, typename...I> requires element_settable<nested_matrix_of_t<Arg&&>, sizeof...(I)>
#else
      template<typename Arg, typename Scalar, typename...I, std::enable_if_t<element_settable<typename nested_matrix_of<Arg&&>::type, sizeof...(I)>, int> = 0>
#endif
      static constexpr Arg&& set(Arg&& arg, const scalar_type_of_t<Arg>& s, I...i)
      {
        set_element(nested_matrix(arg), s, i...);
        return std::forward<Arg>(arg);
      }
    };


    template<typename NestedMatrix>
    struct Dependencies<Eigen3::EigenWrapper<NestedMatrix>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        return make_dense_writable_matrix_from(std::forward<Arg>(arg).nested_matrix());
      }
    };


    template<typename NestedMatrix>
    struct SingleConstant<Eigen3::EigenWrapper<NestedMatrix>> : SingleConstant<std::decay_t<NestedMatrix>>
    {
      SingleConstant(const Eigen3::EigenWrapper<NestedMatrix>& xpr) :
        SingleConstant<std::decay_t<NestedMatrix>> {xpr.nested_matrix()} {};
    };


    template<typename NestedMatrix>
    struct TriangularTraits<Eigen3::EigenWrapper<NestedMatrix>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;
    };


#ifdef __cpp_concepts
    template<hermitian_matrix<Likelihood::maybe> NestedMatrix>
    struct HermitianTraits<Eigen3::EigenWrapper<NestedMatrix>>
#else
    template<typename NestedMatrix>
    struct HermitianTraits<Eigen3::EigenWrapper<NestedMatrix>, std::enable_if_t<hermitian_matrix<NestedMatrix, Likelihood::maybe>>>
#endif
    {
      static constexpr bool is_hermitian = true;
    };


    template<typename NestedMatrix>
    struct Conversions<Eigen3::EigenWrapper<NestedMatrix>>
    {
      template<typename Arg>
      static constexpr decltype(auto) to_diagonal(Arg&& arg) { return OpenKalman::to_diagonal(nested_matrix(std::forward<Arg>(arg))); }

      template<typename Arg>
      static constexpr decltype(auto) diagonal_of(Arg&& arg) { return OpenKalman::diagonal_of(nested_matrix(std::forward<Arg>(arg))); }
    };

  } // namespace interface


} // namespace OpenKalman


namespace Eigen::internal
{
  // ------------------------- //
  //  Eigen::internal::traits  //
  // ------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct EigenWrapperTraits
    {
    private:

      using ElementRef = decltype(OpenKalman::get_element(std::declval<T&>(), 0, 0));
      static constexpr bool lvalue_get_element = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>;

    public:

      using Scalar = scalar_type_of_t<T>;
      using StorageIndex = Index;
      using StorageKind = Dense;
      using XprKind = std::conditional_t<Eigen3::native_eigen_array<T>, ArrayXpr, MatrixXpr>;
      enum {
        RowsAtCompileTime = dynamic_dimension<T, 0> ? Eigen::Dynamic : static_cast<Index>(index_dimension_of_v<T, 0>),
        ColsAtCompileTime = dynamic_dimension<T, 1> ? Eigen::Dynamic : static_cast<Index>(index_dimension_of_v<T, 1>),
        MaxRowsAtCompileTime [[maybe_unused]] = RowsAtCompileTime,
        MaxColsAtCompileTime [[maybe_unused]] = ColsAtCompileTime,
        Flags = (lvalue_get_element ? LvalueBit : 0x0),
      };
    };


#ifdef __cpp_concepts
    template<OpenKalman::Eigen3::has_eigen_traits T>
    struct EigenWrapperTraits<T>
#else
    template<typename T>
    struct EigenWrapperTraits<T, std::enable_if_t<OpenKalman::Eigen3::has_eigen_traits<T>>>
#endif
      : traits<T>
    {
      using StorageKind = Dense;
      enum {
        Flags = std::decay_t<T>::Flags & ~(NestByRefBit),
      };
    };

  } // namespace detail


  template<typename T>
  struct traits<OpenKalman::Eigen3::EigenWrapper<T>> : detail::EigenWrapperTraits<std::decay_t<T>> {};


  // ---------------------------- //
  //  Eigen::internal::evaluator  //
  // ---------------------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename ArgType>
#else
    template<typename ArgType, typename = void>
#endif
    struct EigenWrapperEvaluator : evaluator_base<OpenKalman::Eigen3::EigenWrapper<ArgType>>
    {
    private:

      using XprType = std::decay_t<ArgType>;
      using ElementRef = decltype(OpenKalman::get_element(std::declval<XprType&>(), 0, 0));
      static constexpr bool lvalue_get_element = std::is_same_v<ElementRef, std::decay_t<ElementRef>&>;
      static constexpr bool nested_coeffRef = ((traits<XprType>::Flags & LvalueBit) != 0);
      using my_traits = traits<OpenKalman::Eigen3::EigenWrapper<ArgType>>;

    public:

      explicit EigenWrapperEvaluator(const XprType& t) : m_xpr {const_cast<XprType&>(t)} {}

#ifdef __cpp_concepts
      auto& coeffRef(Index row, Index col) requires nested_coeffRef or lvalue_get_element
#else
      template<bool nc = nested_coeffRef, std::enable_if_t<nc or lvalue_get_element, int> = 0>
      auto& coeffRef(Index row, Index col)
#endif
      {
        if constexpr (nested_coeffRef) return m_xpr.coeffRef(row, col);
        else return OpenKalman::get_element(m_xpr, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }


      constexpr decltype(auto) coeff(Index row, Index col) const
      {
        return OpenKalman::get_element(m_xpr, static_cast<std::size_t>(row), static_cast<std::size_t>(col));
      }

      enum {
        CoeffReadCost = 0,
        Flags = NoPreferredStorageOrderBit,
        Alignment = AlignedMax
      };

    protected:

      XprType& m_xpr;
    };


#ifdef __cpp_concepts
    template<OpenKalman::Eigen3::has_eigen_evaluator ArgType>
    struct EigenWrapperEvaluator<ArgType>
#else
    template<typename ArgType>
    struct EigenWrapperEvaluator<ArgType, std::enable_if_t<OpenKalman::Eigen3::has_eigen_evaluator<ArgType>>>
#endif
      : evaluator<std::decay_t<ArgType>>
    {
      using XprType = std::decay_t<ArgType>;
      using Base = evaluator<XprType>;
      explicit EigenWrapperEvaluator(const XprType& t) : Base {t} {}
    };

  } // namespace detail


  template<typename ArgType>
  struct unary_evaluator<OpenKalman::Eigen3::EigenWrapper<ArgType>> : detail::EigenWrapperEvaluator<ArgType>
  {
    using XprType = OpenKalman::Eigen3::EigenWrapper<ArgType>;
    using Base = detail::EigenWrapperEvaluator<ArgType>;
    explicit unary_evaluator(const XprType& t) : Base {t.nested_matrix()} {}
  };

} // Eigen::internal

#endif //OPENKALMAN_EIGENWRAPPER_HPP
